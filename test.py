#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:08:50 2025

@author: leberac
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, exp, sqrt, log10
from numpy.fft import fft, ifft

# Bridge パターン
# 実装部分：描画方法
class PlotAPI:
    def draw_graph(self,ax, signal: [], fs: int):
        pass

class PlotAPI_FL(PlotAPI):
    def draw_graph(self, ax, signal: [], fs: int):
        N = len(signal)

        # 横軸：周波数、縦軸：レベル表示の図
        if ax==None:
            _, ax = plt.subplots(1,1)
        
        x_axis = np.arange(0, fs/2)
        fft_signal = np.fft.fft(signal, N)
        
        P1 = fft_signal[0:int(fs/2)] / N
        P2 = 10*log10(2 * (abs(P1) ** 2))
        
        power = P2
        
        ax.plot(x_axis, power)
        ax.set_ylabel("Level [dB]")
        ax.set_xlabel("Frequency [Hz]")
        ax.grid()
        
class PlotAPI_TL(PlotAPI):
    def draw_graph(self, ax, signal: [], fs: int):
        N = len(signal)

        # 横軸：周波数、縦軸：レベル表示の図        
        x_axis = np.arange(N) / fs
        
        ax.plot(x_axis, signal)
        ax.set_ylabel("Amplitude [V]")
        ax.set_xlabel("Time [s]")
        ax.grid()
        
# 抽象部分：各信号
class SignalType():
    def __init__(self):
        pass

    def draw(self, ax, plot_api: PlotAPI):
        temp_sig = self.generate()
        plot_api.draw_graph(ax, temp_sig, self.fs)

    def generate(self, count=0):
        # 信号生成
        pass

class NarrowBand(SignalType):
    def __init__(self, f, rms, tau, fs, N):
        super().__init__()
        self.f = f
        self.rms = rms # 
        self.tau = tau # 遅延時間
        self.fs = fs
        self.N = N
        self.t = np.arange(N)/self.fs
        
    def generate(self, count=0):
        st = (self.N *  count + 1) / self.fs
        en =  self.N * (count + 1) / self.fs
        self.t = np.linspace(st, en, num=self.N)
        
        sig = exp(1j*2*pi*self.f*(self.t-self.tau))
        
        return sqrt(2) * self.rms * sig.real

class Noise(SignalType):
    def __init__(self, rms, fs, N, nCh = 1):
        super().__init__()
        self.rms = rms
        self.N = N
        self.fs = fs
        self.nCh = nCh
        
    def generate(self, count=0):
        return self.rms * np.random.randn(self.nCh, self.N)

class ArraySignal(SignalType):
    def __init__(self, pos, c, fs, N):
        super().__init__()
        self.count = 0
        self.pos = pos
        self.signals = []
        self.nCh = pos.shape[0]
        self.c = c
        self.fs = fs
        self.numSignal = 0
        self.N = N
    
    def add(self, sig, theta=0.0, phi=0.0):
        IMP = self._make_delay_filter(theta, phi)
        self.signals.append([sig, theta, phi, IMP])
        self.numSignal += 1
    
    def draw(self, ax, plot_api: PlotAPI):
        temp_sig = self.generate()
        plot_api.draw_graph(ax, temp_sig[0,:], self.fs)
    
    
    def generate(self):
        out = np.zeros([self.numSignal,self.nCh, self.N])
        for i in range(self.numSignal):
            signal = fft(self.signals[i][0].generate(self.count))
            delay_filter = self.signals[i][3]
            out[i] = ifft(signal * delay_filter,n=self.N, axis=1).real
        
        out2 = out.sum(0)
        self.count += 1        
        return out2
    
    def _get_delay(self, theta, phi):
        _theta = theta * pi / 180
        _phi = phi * pi / 180
        s = np.array([cos(_phi) * cos(_theta),
                      cos(_phi) * sin(_theta),
                      sin(_phi)])
        tau = self.pos.dot(s)/self.c
        return tau
    
    def _make_delay_filter(self, theta, phi):
        tau = self._get_delay(theta, phi)
        f = np.linspace(0, int(self.fs/2), int(self.N/2), endpoint=False)

        delay_filter = []
        
        for t in tau:
            phase = exp(1j*2*pi*f*t)
            phase = np.hstack([phase, 0, np.flipud(phase[1:].conj())])
            delay_filter.append(phase)
        
        return np.array(delay_filter)
            
            
# 抽象部分：アレイに入力される信号の集合体
if __name__ == '__main__':
    print("test")
    
    # 共通設定
    c = 1500
    fs = 1024
    N = 1024 # FFT次数 & 1ブロック内の処理点数
    dF = fs/N # 分析幅
    rate = fs / N # 処理レート(1秒毎の処理ブロック数)
    beam = 180
    
    SL = lambda _SL:10 ** (_SL / 20)
    NL = lambda _NL,fs: (10**(_NL/20)) * sqrt(fs/2)
    
    # アレイ
    nCh = 5
    fd = 1000
    d = c/fd*0.5
    pos = np.zeros([nCh, 3])
    x = -1*np.arange(nCh)*d
    pos[:,0] = x - x.mean(0)
    
    # 信号1：狭帯域信号
    f1 = 256
    rms1 = SL(0.0)
    tau1 = 0.0
    a1 = NarrowBand(f1, rms1, tau1, fs, N)

    # 信号2: 白色雑音   
    rms2 = NL(-25,fs)
    a2 = Noise(rms2, fs, N, nCh)
    
    linearray = ArraySignal(pos, c, fs, N)
    linearray.add(a1, 80, 0)
    linearray.add(a2)
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    linearray.draw(ax1, PlotAPI_TL())
    linearray.draw(ax2, PlotAPI_FL())
    
    plt.show()