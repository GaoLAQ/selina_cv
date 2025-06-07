# Selina_CV

# Black-Scholes Model for FTSE 100 Options Pricing - Research Project

# 第一阶段：理论基础与公式推导（1-2周）

## 1. Black-Scholes模型的基本假设

理解模型的基本假设是分析其局限性的关键：

- **标的资产价格**：服从对数正态分布  
  `dS = μSdt + σSdX`

- **无风险利率**：已知且恒定（r = constant）

- **股息政策**：  
  - 基础模型假设标的资产不支付股息  
  - 可扩展至含股息情况（q为股息率）

- **市场环境**：  
  ✅ 无摩擦市场（无交易成本、税收等）  
  ✅ 允许无限制卖空  
  ✅ 无套利机会

- **期权类型**：欧式期权（仅到期日可行权）

- **波动率**：恒定（σ = constant）

## 2. Black-Scholes公式的数学推导

### 偏微分方程推导步骤

1. **资产价格动态**  
   几何布朗运动：  
   ```math
   dS = \mu S dt + \sigma S dX
   ```
2. **构建对冲组合**
   ```math
   \Pi = V - \Delta S
   ```
3. **应用伊藤引理**
   对期权价格V(S,t)展开：
   ```math
   dV = \frac{\partial V}{\partial t}dt + \frac{\partial V}{\partial S}dS + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2}dt
   ```
4. **消除随机项**
   通过选择
   ```math
   Δ = ∂V/∂S消除dX项
   ```
5. **无风险组合条件**
   ```math
   d\Pi = r\Pi dt
   ```
6.  **得到Black-Scholes PDE**
   ```math
   \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2} + rS
   ```
   ```math
   \frac{\partial V}{\partial S} - rV = 0
   ```
# 第二阶段：编程实现（2-3周）
/ React 建立
/ Python 建立
/ Database建立

# 技术栈构建方案

## React 前端框架建立

### 核心特性
```javascript
// 典型React组件示例
import React, { useState, useEffect } from 'react';

function App() {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData);
  }, []);

  return (
    <div className="App">
      {data.map(item => (
        <Card key={item.id} data={item} />
      ))}
    </div>
  );
}
  ```python
  def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
  ```
# 第三阶段：实证分析（2周）

### 1. 分析框架

#### 1.1 数据预处理
```python
# 示例代码：数据清洗
import pandas as pd

def clean_data(df):
    """处理缺失值和异常值"""
    df = df.dropna()
    df = df[(df['implied_vol'] > 0) & (df['implied_vol'] < 1)]  # 过滤不合理波动率
    return df
```
## 1.2 波动率计算

### 波动率计算方法对比

| **波动率类型** | **计算方法** | **数学表达** | **Python实现** |
|----------------|--------------|--------------|----------------|
| **历史波动率** | 30日滚动标准差 | ```math \sigma_{hist} = \text{std}(\ln(\frac{S_t}{S_{t-1}})) \times \sqrt{252} ``` | ```python returns.rolling(30).std() * np.sqrt(252)``` |
| **隐含波动率** | 牛顿迭代法反解 | ```math \text{Find } \sigma \text{ s.t. } BS(S,K,T,r,\sigma) = C_{market} ``` | 见下方代码实现 |

### 隐含波动率计算实现

```python
from scipy.stats import norm

def implied_vol(S, K, T, r, C_market, tol=0.0001, max_iter=100):
    """
    使用牛顿法计算隐含波动率
    
    参数:
        S: 标的资产价格
        K: 执行价格
        T: 到期时间(年)
        r: 无风险利率
        C_market: 市场观察到的期权价格
        tol: 容忍误差
        max_iter: 最大迭代次数
        
    返回:
        隐含波动率
    """
    sigma = 0.3  # 初始猜测值
    
    for i in range(max_iter):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        C_est = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)  # Vega值
        
        diff = C_est - C_market
        if abs(diff) < tol:
            return sigma
            
        sigma = sigma - diff/vega
    
    return sigma  # 未收敛时返回最后估计值
```
## 2.2 波动率微笑分析

### 波动率微笑分析实现

#### Python可视化代码
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_volatility_smile(df, title="FTSE 100期权波动率微笑"):
    """
    绘制波动率微笑曲线
    
    参数:
        df: 包含以下列的DataFrame:
            - moneyness: 标的价格/执行价 (S/K)
            - implied_vol: 隐含波动率
            - maturity: 到期天数
        title: 图表标题
    """
    plt.figure(figsize=(12, 7))
    
    # 按到期日分组绘制
    for maturity in sorted(df['maturity'].unique()):
        subset = df[df['maturity'] == maturity]
        plt.scatter(
            subset['moneyness'], 
            subset['implied_vol'],
            label=f'{maturity}天到期',
            s=50,  # 点大小
            alpha=0.7
        )
    
    # 图表美化
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)  # 平价点参考线
    plt.title(title, fontsize=14)
    plt.xlabel('Moneyness (S/K)', fontsize=12)
    plt.ylabel('Implied Volatility (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='到期期限', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加波动率微笑的典型解释注释
    plt.annotate('虚值看跌期权(左端)\n通常有更高波动率',
                xy=(0.8, df['implied_vol'].max()*0.9),
                xytext=(0.7, df['implied_vol'].max()*0.7),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.show()
```
## 敏感性分析

#### 分析方法
```python
def analyze_maturity_effect(df):
    """按到期时间分组分析定价误差"""
    bins = [0, 30, 90, 365]  # 短期/中期/长期
    labels = ['短期(<30天)', '中期(30-90天)', '长期(>90天)']
    df['maturity_group'] = pd.cut(df['days_to_maturity'], bins=bins, labels=labels)
    
    return df.groupby('maturity_group')['pricing_error'].agg(
        ['mean', 'std', 'count']
    ).rename(columns={
        'mean': '平均绝对误差(MAE)',
        'std': '误差标准差'
    })
```
