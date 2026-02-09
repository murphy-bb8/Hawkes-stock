import math
import numpy as np


class MHP:
    """
    Simple 1D Hawkes process simulator with optional sine exogenous term.

    Intensity: lambda(t) = mu + alpha * r(t) + beta0 * sin(beta1 * t)
    where r(t) = sum_{s < t} exp(-beta * (t - s)).

    Parameters
    ----------
    lambda_0 : float
        Baseline intensity mu >= 0
    alpha : float
        Excitation strength >= 0
    beta : float
        Exponential decay rate > 0
    beta0 : float
        Sine amplitude (can be negative)
    beta1 : float
        Sine frequency (rad / time), > 0 recommended
    """

    def __init__(self, lambda_0: float, alpha: float, beta: float, beta0: float = 0.0, beta1: float = 1.0):
        self.mu = float(lambda_0)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)

    def simulate(self, T: float = None, target_events: int = None, seed: int = 0, max_events: int = 2_000_000):
        """Ogata thinning with adaptive upper bound.

        - If `target_events` is given, simulate until that many events occur
          (or until max_events guard triggers). The horizon T is then the
          last event time.
        - Else, simulate up to time T.
        """
        rng = np.random.RandomState(seed)
        t = 0.0
        r = 0.0
        out = []
        use_target = target_events is not None
        n_target = int(target_events) if use_target else None

        while True:
            if use_target:
                if len(out) >= n_target or len(out) >= max_events:
                    break
            else:
                if T is None:
                    raise ValueError("Provide T or target_events")
                if t >= T or len(out) >= max_events:
                    break

            # Upper bound using |sin| <= 1
            lam_bar = self.mu + abs(self.beta0) + self.alpha * r
            if lam_bar <= 1e-12:
                # no more arrivals possible with current bound
                break
            dt = rng.exponential(1.0 / lam_bar)
            t_next = t + dt
            if (not use_target) and (t_next > T):
                break

            # decay memory
            decay = math.exp(-self.beta * dt)
            r *= decay

            lam = self.mu + self.alpha * r + self.beta0 * math.sin(self.beta1 * t_next)
            lam = max(lam, 0.0)
            if rng.rand() * lam_bar <= lam:
                out.append(t_next)
                r += 1.0
            t = t_next

        return np.asarray(out, dtype=float)

import numpy as np

class MHP:
    def __init__(self, lambda_0, alpha, beta, T):
        """
        初始化 MHP 模型
        :param lambda_0: 背景强度
        :param alpha: 激励参数
        :param beta: 衰减参数
        :param T: 模拟的总时间
        """
        self.lambda_0 = lambda_0  # 背景强度
        self.alpha = alpha  # 激励参数
        self.beta = beta  # 衰减参数
        self.T = T  # 模拟时间
        self.times = []  # 事件发生的时间列表

    def update_intensity(self, t):
        """
        根据历史事件更新强度
        :param t: 当前时间
        :return: 当前的强度值
        """
        intensity = self.lambda_0
        for prev_t in self.times:
            if t > prev_t:  # 确保时间顺序正确
                intensity += self.alpha * np.exp(-self.beta * (t - prev_t))
        return max(0, intensity)  # 确保强度非负

    def simulate(self):
        """
        使用 Ogata 改进的薄化算法模拟 Hawkes 过程
        :return: 事件发生的时间列表
        """
        t = 0
        self.times = []  # 重置事件列表
        
        while t < self.T:
            # 计算当前强度
            current_intensity = self.update_intensity(t)
            
            # 生成下一个潜在事件的时间间隔
            if current_intensity > 0:
                dt = np.random.exponential(1 / current_intensity)
                t += dt
                
                if t >= self.T:
                    break
                    
                # 计算新时间点的强度
                new_intensity = self.update_intensity(t)
                
                # Ogata 薄化算法：以概率 new_intensity/current_intensity 接受事件
                u = np.random.uniform(0, 1)
                if u <= new_intensity / current_intensity:
                    self.times.append(t)
            else:
                # 如果强度为0，直接跳到时间T
                t = self.T
                
        return self.times

    def debug_simulation(self, max_events=1000):
        """
        调试版本的模拟，添加详细的输出信息
        """
        print("开始调试模拟...")
        print(f"参数: lambda_0={self.lambda_0}, alpha={self.alpha}, beta={self.beta}, T={self.T}")
        
        t = 0
        self.times = []
        event_count = 0
        
        while t < self.T and event_count < max_events:
            current_intensity = self.update_intensity(t)
            print(f"时间 t={t:.3f}, 当前强度={current_intensity:.3f}, 事件数={len(self.times)}")
            
            if current_intensity > 0:
                dt = np.random.exponential(1 / current_intensity)
                t += dt
                
                if t >= self.T:
                    break
                    
                new_intensity = self.update_intensity(t)
                u = np.random.uniform(0, 1)
                accept_prob = new_intensity / current_intensity if current_intensity > 0 else 0
                
                print(f"  新时间 t={t:.3f}, 新强度={new_intensity:.3f}, 接受概率={accept_prob:.3f}, u={u:.3f}")
                
                if u <= accept_prob:
                    self.times.append(t)
                    event_count += 1
                    print(f"  ✓ 接受事件 {event_count} 在时间 {t:.3f}")
                else:
                    print(f"  ✗ 拒绝事件")
            else:
                print("  强度为0，结束模拟")
                t = self.T
                
        print(f"模拟完成，共生成 {len(self.times)} 个事件")
        return self.times

    def plot_events(self):
        """
        绘制事件时间的直方图
        """
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.times, bins=50, alpha=0.7, color='g', edgecolor='black')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('事件计数', fontsize=12)
        plt.title('Hawkes过程事件时间分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()