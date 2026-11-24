from __future__ import annotations

import numpy as np
import pytest

from python.window_projection import project_fifo_only, decode_fifo


def obs(t=0.0, window=0, rate=2500.0, batches=()):
    a = np.zeros(1519, dtype=np.float32)
    a[0] = np.float32(t/28800.0)
    a[2] = np.float32(window/4.0)
    a[17] = float(window == 4)
    a[1304] = np.float32(rate/2500.0)
    a[1300] = np.float32(sum(q for q,_ in batches)/500.0)
    for i,(q,g) in enumerate(batches):
        k=1391+3*i
        a[k] = 1
        a[k+1] = np.float32(q/500.0)
        a[k+2] = np.float32(g)
    return a


def test_current_and_next_and_mass_metal_conservation():
    # 2500 t/h = 0.6944 t/s. At t=7100, only 69.444 t can be
    # processed before window boundary; the remaining 130.556 t -> next.
    p=project_fifo_only(obs(7100,0,batches=[(100,.2),(100,.4)]))
    a=100*2500/3600
    assert p.produced_tons_by_window[0] == pytest.approx(a,abs=0.0005)
    assert p.produced_tons_by_window[1] == pytest.approx(200-a,abs=0.0005)
    assert p.produced_metal_by_window[0] == pytest.approx(a*.2,abs=0.0005)
    assert sum(p.produced_tons_by_window) == pytest.approx(200,abs=0.0005)
    assert sum(p.produced_metal_by_window) == pytest.approx(60,abs=0.0005)
    assert p.features[2] == 1 and p.features[5] == 1


def test_empty_fifo_is_valid_zero_contribution():
    z=project_fifo_only(obs(100,0)).features
    assert z.tolist() == [0,0,1,0,0,1]


def test_final_window_next_invalid():
    z=project_fifo_only(obs(24000,3,batches=[(250,.26)])).features
    assert z[2] == 1
    assert list(z[3:]) == [0,0,0]


def test_terminal_both_invalid():
    z=project_fifo_only(obs(28800,4,batches=[(100,.3)])).features
    assert z.tolist() == [0]*6


def test_strict_fifo_mass_audit():
    x=obs(100,0,batches=[(150,.3)])
    x[1300]=0.0
    with pytest.raises(ValueError,match="FIFO mass"):
        decode_fifo(x)


def test_fifo_no_hole():
    x=obs(0,0,batches=[(100,.2)])
    x[1391]=0; x[1392]=0; x[1393]=0
    x[1394]=1; x[1395]=.2; x[1396]=.2
    with pytest.raises(ValueError,match="after empty"):
        decode_fifo(x,check_buffer=False)


def test_window_boundary_exact():
    p=project_fifo_only(obs(7200,1,batches=[(500,.3)]))
    assert p.produced_tons_by_window[0] == 0
    assert p.produced_tons_by_window[1] == pytest.approx(500)
