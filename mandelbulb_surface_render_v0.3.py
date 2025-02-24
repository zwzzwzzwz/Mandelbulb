# -*- coding: utf-8 -*-
"""
创建时间：2025年2月24日 04:03:56
作者：粥当当
"""

import numpy as np
import torch
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import tempfile
from tqdm import tqdm
from OpenGL.arrays import vbo
from OpenGL.GL.shaders import compileProgram, compileShader

# Mandelbulb 常量定义
POWER, MAX_ITER, THRESHOLD = 8, 60, 1.0  # 幂次、最大迭代次数、逃逸阈值
RESOLUTION, BLOCK_SIZE = 5000, 100  # 分辨率和分块大小
THRESHOLD_SQ = THRESHOLD * THRESHOLD  # 阈值的平方，用于加速计算

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 顶点着色器（带动态点大小）
vertex_shader = """
#version 330
in vec3 position;
in vec3 color;
out vec3 fragColor;
out float depth;
uniform mat4 modelViewProj;
void main() {
    gl_Position = modelViewProj * vec4(position, 1.0);
    fragColor = color;
    depth = gl_Position.z;
    gl_PointSize = 5.0 - depth * 0.1;
}
"""

# 片段着色器（带透明度）
fragment_shader = """
#version 330
in vec3 fragColor;
in float depth;
out vec4 outColor;
void main() {
    float alpha = clamp(1.0 - depth * 0.05, 0.2, 1.0);
    outColor = vec4(fragColor, alpha);
}
"""

def setup_shader():
    """编译并返回着色器程序"""
    return compileProgram(
        compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    )

def draw_points_vbo_shader(points_vbos, colors_vbos, shader, scale=1.0):
    """使用 VBO 和着色器绘制点云"""
    glUseProgram(shader)
    glPushMatrix()
    glScalef(scale, scale, scale)
    
    mvp = glGetFloatv(GL_MODELVIEW_MATRIX) @ glGetFloatv(GL_PROJECTION_MATRIX)
    glUniformMatrix4fv(glGetUniformLocation(shader, "modelViewProj"), 1, GL_FALSE, mvp)

    for points_vbo, colors_vbo in zip(points_vbos, colors_vbos):
        points_vbo.bind()
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        colors_vbo.bind()
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glDrawArrays(GL_POINTS, 0, len(points_vbo))

        points_vbo.unbind()
        colors_vbo.unbind()
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    glPopMatrix()
    glUseProgram(0)

def mandelbulb_point(coords):
    """计算 Mandelbulb 的点迭代值"""
    coords = coords.to(torch.float32)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0, y0, z0 = x.clone(), y.clone(), z.clone()  # 保存初始坐标
    values = torch.zeros(coords.shape[0], device=device, dtype=torch.float32)
    active = torch.ones(coords.shape[0], dtype=torch.bool, device=device)

    # 迭代计算 Mandelbulb
    for i in range(MAX_ITER):
        r_sq = x*x + y*y + z*z  # 计算距离平方
        r = torch.sqrt(r_sq)    # 距离
        theta = torch.atan2(r, z)  # 极角
        phi = torch.atan2(y, x)    # 方位角
        
        # 球坐标系下的幂次变换
        r_pow = r**POWER
        theta_pow = theta * POWER
        sin_theta_pow = torch.sin(theta_pow)
        
        x_new = r_pow * sin_theta_pow * torch.cos(phi * POWER) + x0
        y_new = r_pow * sin_theta_pow * torch.sin(phi * POWER) + y0
        z_new = r_pow * torch.cos(theta_pow) + z0

        # 仅更新仍在逃逸范围内的点
        x = torch.where(active, x_new, x)
        y = torch.where(active, y_new, y)
        z = torch.where(active, z_new, z)

        # 检查逃逸条件
        r_sq = x*x + y*y + z*z
        new_active = r_sq <= THRESHOLD_SQ
        just_escaped = active & ~new_active
        values[just_escaped] = i + 1
        active = new_active

        if not active.any():
            break

    values[active] = MAX_ITER
    return values / MAX_ITER, active

def generate_mandelbulb(res, block_size):
    """生成 Mandelbulb 的表面点和颜色"""
    temp_dir = tempfile.mkdtemp(dir="D:\\SYSTemp")
    points_list, colors_list = [], []

    grid = torch.linspace(-3.0, 3.0, res, device=device)
    step = (grid[1] - grid[0]).item()

    # 分块处理以优化内存使用
    for i in tqdm(range(0, res, block_size), desc="生成分块"):
        for j in range(0, res, block_size):
            for k in range(0, res, block_size):
                i_end, j_end, k_end = min(i + block_size, res), min(j + block_size, res), min(k + block_size, res)
                X, Y, Z = torch.meshgrid(grid[i:i_end], grid[j:j_end], grid[k:k_end], indexing='ij')
                coords = torch.stack((X, Y, Z), dim=-1).reshape(-1, 3)

                # 提前过滤远离 Mandelbulb 的点
                r_sq_initial = coords.pow(2).sum(dim=1)
                mask = r_sq_initial < 9.0  # 仅计算半径小于3的点
                if not mask.any():
                    continue
                coords = coords[mask]

                values, active = mandelbulb_point(coords)
                
                # 优化表面检测：保留接近边界的点
                r_sq = coords.pow(2).sum(dim=1)
                surface_mask = active & (r_sq > 0.5 * THRESHOLD_SQ) & (r_sq < THRESHOLD_SQ * 1.5)
                neighbor = coords + torch.tensor([step, 0, 0], device=device)
                _, neighbor_active = mandelbulb_point(neighbor)
                surface_mask |= (~active & neighbor_active & (values > 0.1))

                if surface_mask.any():
                    points_list.append(coords[surface_mask].cpu().numpy())
                    colors_list.append(values[surface_mask].cpu().numpy())

    points = np.concatenate(points_list) if points_list else np.array([])
    colors = np.concatenate(colors_list) if colors_list else np.array([])

    print(f"生成了 {len(points)} 个表面点")
    if points.size:
        np.savez(os.path.join(temp_dir, "mandelbulb.npz"), points=points, colors=colors)

    return temp_dir, points, colors

def init_opengl(width, height):
    """初始化 OpenGL 环境"""
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    glClearColor(0, 0, 0, 1.0)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(60, width/height, 0.1, 100.0)
    glTranslatef(0.0, 0.0, -40.0)

    # 设置基本光照
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glLightfv(GL_LIGHT0, GL_POSITION, [10.0, 10.0, -10.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.2, 1.2, 1.2, 1.0])

def setup_vbo(points, colors, chunk_size=50000):
    """设置顶点缓冲对象 (VBO)"""
    if points.shape[0] != colors.shape[0] or points.shape[1] != 3:
        raise ValueError("点和颜色的维度不匹配")

    # 根据迭代值映射颜色
    colors_rgb = np.zeros((len(colors), 3), dtype='float32')
    iter_values = colors * MAX_ITER
    third, two_third = MAX_ITER / 3.0, MAX_ITER * 2.0 / 3.0
    norm_val = iter_values / MAX_ITER

    colors_rgb[:, 0] = np.minimum(np.select(
        [iter_values < third, iter_values < two_third], [norm_val * 3, 0], 0), 1.1)
    colors_rgb[:, 1] = np.minimum(np.select(
        [iter_values < third, iter_values < two_third], [0, norm_val * 3], 0), 1.1)
    colors_rgb[:, 2] = np.minimum(np.select(
        [iter_values < two_third, iter_values < MAX_ITER], [0, norm_val * 3], 0), 1.1)

    # 分块创建 VBO
    points_vbos, colors_vbos = [], []
    for i in range(0, len(points), chunk_size):
        chunk_points = np.ascontiguousarray(points[i:i + chunk_size], dtype='float32')
        chunk_colors = np.ascontiguousarray(colors_rgb[i:i + chunk_size], dtype='float32')
        points_vbos.append(vbo.VBO(chunk_points))
        colors_vbos.append(vbo.VBO(chunk_colors))

    return points_vbos, colors_vbos

def draw_axes(scale=1.0):
    """绘制坐标轴"""
    glPushMatrix()
    glScalef(scale, scale, scale)
    glLineWidth(2.0)

    glBegin(GL_LINES)
    for color, axis in [((1.0, 0.0, 0.0), (6.0, 0.0, 0.0)), ((0.0, 1.0, 0.0), (0.0, 6.0, 0.0)), 
                        ((0.0, 0.0, 1.0), (0.0, 0.0, 6.0))]:
        glColor3f(*color)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(*axis)
    glEnd()

    glPopMatrix()

def main():
    """主函数：生成并渲染 Mandelbulb"""
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    temp_dir, points, colors = generate_mandelbulb(RESOLUTION, BLOCK_SIZE)
    if not points.size:
        return

    points_vbos, colors_vbos = setup_vbo(points, colors)
    init_opengl(1600, 1200)
    shader = setup_shader()
    clock = pygame.time.Clock()

    angle_x, angle_y, scale = 0, 0, 2.0
    mouse_down = False

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                pygame.quit()
                return
            elif event.type == KEYDOWN:
                scale += 0.1 if event.key == K_UP else -0.1 if event.key == K_DOWN else 0
            elif event.type == MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == MOUSEBUTTONUP:
                mouse_down = False
            elif event.type == MOUSEMOTION and mouse_down:
                angle_x += event.rel[1] * 0.5
                angle_y += event.rel[0] * 0.5
            elif event.type == MOUSEWHEEL:
                scale += event.y

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)
        draw_points_vbo_shader(points_vbos, colors_vbos, shader, scale)
        draw_axes(scale)
        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()