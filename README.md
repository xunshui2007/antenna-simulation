# Antenna Simulation

基于 FDTD (时域有限差分) 方法的天线仿真程序。

## 功能

- 2D TMz FDTD 电磁场仿真
- 中心馈电偶极子天线建模
- 高斯脉冲激励
- Mur 一阶吸收边界条件
- 实时 Ez 场分布动画
- 远场辐射方向图计算

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python fdtd_dipole.py
```

## License

MIT

## 网页版

直接打开 `index.html` 即可在浏览器中运行 FDTD 偶极子天线仿真，无需安装任何依赖。

### 网页版功能

- 实时 Ez 场分布彩色云图
- 极坐标辐射方向图
- 运行 / 暂停控制
- 速度调节
- 自适应色阶开关
- 一键重置
