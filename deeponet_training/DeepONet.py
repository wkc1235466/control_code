import os

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from timeit import default_timer
import deepxde as dde

torch.set_default_device("cpu")


def solveIntegralFD(M, a_func, lambda_func, c, return_full=True, compute_derivative=False):
    M = int(M)
    if M < 2:
        raise ValueError("M must be >= 2")

    M2 = 2 * M
    N = M2 + 1
    h = 1.0 / M2

    x = np.linspace(-0.5, 0.5, N)
    y = x.copy()

    a_vals = np.asarray(a_func, dtype=float).reshape(-1)
    lambda_vals = np.asarray(lambda_func, dtype=float).reshape(-1)

    if len(a_vals) != N:
        raise ValueError(f"Invalid a_func length: expected {N}, got {len(a_vals)}")

    if len(lambda_vals) != N:
        raise ValueError(f"Invalid lambda_func length: expected {N}, got {len(lambda_vals)}")

    if np.any(np.abs(a_vals) < 1e-14):
        raise ValueError("a(x) contains values near zero; division is not possible")

    a_prime = np.gradient(a_vals, h, edge_order=2)
    q_vals = lambda_vals + c

    k = np.zeros((N, N), dtype=float)

    for i in range(N):
        j_anti = M2 - i
        if 0 <= j_anti < N:
            k[i, j_anti] = 0.0

    k[M, M] = 0.0

    def ki1j(i, j):
        if j - 1 < 0 or j + 1 >= N:
            return 0.0
        kij = 0.5 * (k[i, j + 1] + k[i, j - 1])
        return (
            (
                (2.0 * (a_vals[i] - a_vals[j]) + h * h * q_vals[j]) * kij
                + a_vals[j + 1] * k[i, j + 1]
                + a_vals[j - 1] * k[i, j - 1]
            ) / a_vals[i]
            - k[i - 1, j]
        )

    for i in range(M, N - 1):
        new_i = i + 1
        j_diag = new_i
        j_anti = M2 - new_i

        if 0 <= j_anti < N:
            k[new_i, j_anti] = 0.0

        denom = 1.0 + h * a_prime[j_diag] / (4.0 * a_vals[j_diag])
        k[new_i, j_diag] = (
            (1.0 - h * a_prime[i] / (4.0 * a_vals[i])) * k[i, i]
            + h / 4.0 * (q_vals[i] / a_vals[i] + q_vals[j_diag] / a_vals[j_diag])
        ) / denom

        j_near_diag = new_i - 1
        if j_near_diag > j_anti and 0 <= j_near_diag < N:
            k[new_i, j_near_diag] = (
                k[i, i]
                - h * a_prime[i] / (4.0 * a_vals[i]) * k[i, i]
                + h * q_vals[i] / (4.0 * a_vals[i])
            )

        j_start = j_anti + 1
        j_end = j_diag - 1
        for j in range(j_start, j_end):
            if j < 0 or j >= N:
                continue
            if j == j_anti or j == j_diag:
                continue
            if i - 1 >= 0 and j - 1 >= 0 and j + 1 < N:
                k[new_i, j] = ki1j(i, j)

    for i in range(M, 0, -1):
        new_i = i - 1
        j_diag = new_i
        j_anti = M2 - new_i

        if 0 <= j_anti < N:
            k[new_i, j_anti] = 0.0

        denom_left = 1.0 - h * a_prime[j_diag] / (4.0 * a_vals[j_diag])
        k[new_i, j_diag] = (
            (1.0 + h * a_prime[i] / (4.0 * a_vals[i])) * k[i, i]
            - h / 4.0 * (q_vals[i] / a_vals[i] + q_vals[j_diag] / a_vals[j_diag])
        ) / denom_left

        j_near_diag = i
        if j_diag < j_near_diag < j_anti and 0 <= j_near_diag < N:
            k[new_i, j_near_diag] = (
                k[i, i]
                + h * a_prime[i] / (4.0 * a_vals[i]) * k[i, i]
                - h * q_vals[i] / (4.0 * a_vals[i])
            )

        j_start = i + 1
        j_end = j_anti
        for j in range(j_start, j_end):
            if j < 0 or j >= N:
                continue
            if j == j_diag or j == j_anti:
                continue
            if i + 1 < N and j - 1 >= 0 and j + 1 < N:
                k[new_i, j] = (
                    (
                        (2.0 * (a_vals[i] - a_vals[j]) + h * h * q_vals[j])
                        * 0.5 * (k[i, j + 1] + k[i, j - 1])
                        + a_vals[j + 1] * k[i, j + 1]
                        + a_vals[j - 1] * k[i, j - 1]
                    ) / a_vals[i]
                    - k[i + 1, j]
                )

    for i in range(N):
        j_anti = M2 - i
        if 0 <= j_anti < N:
            k[i, j_anti] = 0.0

    k[M, M] = 0.0

    k_out = k if return_full else k[::2, ::2]
    if not compute_derivative:
        return k_out
    dk_05, dk_m05 = boundary_derivatives(k, verbose=False)
    if not return_full:
        dk_05, dk_m05 = dk_05[::2], dk_m05[::2]
    return k_out, dk_05, dk_m05


def _column_spline_dx(vals, y, side, margin=6):
    N = len(y)
    out = np.full(N, np.nan)
    for j, xj in enumerate(y):
        lo = abs(xj)
        m = (y >= lo - 1e-12) if side > 0 else (y <= -lo + 1e-12)
        xs, col = y[m], vals[m, j]
        if xs.size < margin + 3 or (xs[-1] - xs[0]) < margin * (xs[1] - xs[0]):
            continue
        out[j] = CubicSpline(xs, col, bc_type="natural")(0.5 if side > 0 else -0.5, 1)
    return out


def _row_fallback(vals, y, side):
    N = len(y)
    h = 0.5 / (N - 1) * 2
    h = y[1] - y[0]
    out = np.full(N, np.nan)
    for j in range(N):
        if abs(y[j]) > 0.5 - 2 * h + 1e-12:
            continue
        if side > 0:
            out[j] = (3 * vals[-1, j] - 4 * vals[-2, j] + vals[-3, j]) / (2 * h)
        else:
            out[j] = (-3 * vals[0, j] + 4 * vals[1, j] - vals[2, j]) / (2 * h)
    return out


def _extrapolate_ends(vals, y):
    out = np.array(vals, dtype=float, copy=True)
    good = np.flatnonzero(np.isfinite(out))
    if good.size < 3:
        return out
    for j in range(len(out)):
        if np.isfinite(out[j]):
            continue
        near = good[np.argsort(np.abs(good - j))[:3]]
        near = np.sort(near)
        out[j] = float(np.polyval(np.polyfit(y[near], out[near], 1), y[j]))
    return out


def boundary_derivatives(k_full, verbose=False):
    vals = np.asarray(k_full, dtype=float)
    N = vals.shape[1]
    y = np.linspace(-0.5, 0.5, N)
    right = _column_spline_dx(vals, y, +1)
    left = _column_spline_dx(vals, y, -1)
    rr = _row_fallback(vals, y, +1)
    rl = _row_fallback(vals, y, -1)
    right = _extrapolate_ends(np.where(np.isfinite(right), right, rr), y)
    left = _extrapolate_ends(np.where(np.isfinite(left), left, rl), y)
    if verbose:
        print(f'    [info] derivative ranges: +0.5 [{right.min():+.4f}, {right.max():+.4f}]  '
              f'-0.5 [{left.min():+.4f}, {left.max():+.4f}]')
    return right, left


torch.manual_seed(42)
np.random.seed(42)

_sp = np.linspace(-0.5, 0.5, 101)
_a = 0.40 + 1.50 * _sp ** 2
_lb = 0.10 + 0.10 * _sp
_Q = 0.5 * 0.10 - 0.5 * (3.0 * _sp / _a) * _lb - 0.25 * _lb ** 2 / _a
_k = solveIntegralFD(50, _a, _Q, 8.0, return_full=True, compute_derivative=True)[0]
_km = float(np.abs(_k).max())
print(f'Solver self-check: max|k| = {_km:.4f}   (expected 4.7998)')
print('OK: matches the MATLAB implementation; ready to generate data' if abs(_km - 4.7998) < 1e-3
      else 'FAIL: does not match the MATLAB implementation; check the embedded solver')

X = 0.5
dx = 0.01
nx = int(round(X / dx))
spatial = np.linspace(-X, X, 2 * nx + 1, dtype=np.float32)

karr = []
fun1 = []
fun2 = []

lamArr = np.random.uniform(0.7, 1.3, 1000)
lamArr1 = np.random.uniform(1.0, 2.0, 1000)

A0_T, C_T = 0.40, 1.50
L0_T, L1_T = 0.10, 0.10

m = nx
for i in range(1000):
    s = lamArr[i]
    a_vals = s * (A0_T + C_T * spatial ** 2)
    lam_bar = L0_T + (L1_T * s) * spatial

    a_prime = 2.0 * C_T * s * spatial
    lam_bar_prime = L1_T * s
    Q = (0.5 * lam_bar_prime
         - 0.5 * (a_prime / a_vals) * lam_bar
         - 0.25 * lam_bar ** 2 / a_vals)

    mu = 5.0 * lamArr1[i]
    k = solveIntegralFD(m, a_vals, Q, mu)
    if i % 100 == 0:
        print(f'Sample {i}: mu={mu:.3f}  a in [{a_vals.min():.3f},{a_vals.max():.3f}]  '
              f'Q∈[{Q.min():+.4f},{Q.max():+.4f}]  k∈[{k.min():.3f},{k.max():.3f}]')

    karr.append(k)
    fun1.append(a_vals)
    fun2.append(Q)

x_Fun1 = np.array(fun1, dtype=np.float32)
x_Fun2 = np.array(fun2, dtype=np.float32)

grid_size = 2 * nx + 1
x_fun1 = np.repeat(x_Fun1[:, :, np.newaxis], grid_size, axis=2)
x_fun2 = np.repeat(x_Fun2[:, :, np.newaxis], grid_size, axis=2)

AA = np.ones((grid_size, grid_size))
c_val = 5 * lamArr1
x_c_val = c_val[:, np.newaxis, np.newaxis] * AA

x_fun1 = np.array(x_fun1, dtype=np.float32)
x_fun2 = np.array(x_fun2, dtype=np.float32)
x_c_val = np.array(x_c_val, dtype=np.float32)

x_fun1 = x_fun1.reshape(x_fun1.shape[0], -1)
x_fun2 = x_fun2.reshape(x_fun2.shape[0], -1)
x_c_val = x_c_val.reshape(x_c_val.shape[0], -1)

y = np.array(karr, dtype=np.float32).reshape(len(karr), -1)

print(f"Spatial function shape: fun1={x_fun1.shape}, fun2={x_fun2.shape}")
print(f"Constant parameter shape: c_val={x_c_val.shape}")
print(f"Output shape: y={y.shape}")

batch_size = 20

grids = [spatial, spatial]
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = torch.from_numpy(grid).cuda()
print(grid.shape)

x = np.stack((x_fun1, x_fun2, x_c_val), axis=1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

train_gen = torch.Generator()
train_gen.manual_seed(42)
trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                       shuffle=True, generator=train_gen)
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
print("Data load success!")


def count_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn_ = 1
        for s in list(p.size()):
            nn_ = nn_ * s
        pp += nn_
    return pp


class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = torch.nn.Linear(16928, 1028)
        self.fc2 = torch.nn.Linear(1028, 256)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.shape, self.shape))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


dim_x = 2
m = (2 * nx + 1) ** 2

branch = BranchNet(2 * nx + 1)

model = dde.nn.DeepONetCartesianProd([m, branch], [dim_x, 128, 256, 256], "relu", "Glorot normal").cuda()

step_size = 50
gamma = 0.5
learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

print(count_params(model))


def boundary_mse(pred_k, true_k, mask):
    return ((pred_k[:, mask] - true_k[:, mask]) ** 2).mean()


bd_mask = (torch.abs(grid[:, 0] - 0.5) < 1e-2) | (torch.abs(grid[:, 0] + 0.5) < 1e-2)

epochs = 500
loss_fn = torch.nn.MSELoss()
train_lossArr = []
test_lossArr = []
time_Arr = []

print("epoch \ttime \t\t\ttrain_loss \ttrain_data \ttrain_bd \ttest_loss \tbd_weight \tlr")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    train_data_loss = 0
    train_bd_loss = 0

    if ep < 100:
        bd_weight = 0.0
    elif ep < 300:
        bd_weight = 0.1 * min(1.0, (ep - 100) / 200.0)
    else:
        bd_weight = 0.1

    for x_batch, y_batch in trainData:
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        optimizer.zero_grad()
        out = model((x_batch, grid))

        data_loss = loss_fn(out, y_batch)

        if bd_weight > 0:
            bd_loss = boundary_mse(out, y_batch, bd_mask)
            total_loss = data_loss + bd_weight * bd_loss
            train_bd_loss += bd_loss.item()
        else:
            total_loss = data_loss

        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        train_data_loss += data_loss.item()

    scheduler.step()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in testData:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            out = model((x_batch, grid))
            test_loss += loss_fn(out, y_batch).item()

    train_loss /= len(trainData)
    test_loss /= len(testData)
    train_data_loss /= len(trainData)
    train_bd_loss /= len(trainData)

    train_lossArr.append(train_loss)
    test_lossArr.append(test_loss)

    t2 = default_timer()
    time_Arr.append(t2 - t1)

    if ep % 10 == 0:
        print("{} \t{:.2f} \t{:.6f} \t{:.6f} \t{:.6f} \t{:.6f} \t{:.4f} \t{:.2e}".format(
            ep, t2 - t1, train_loss, train_data_loss, train_bd_loss, test_loss, bd_weight,
            optimizer.state_dict()['param_groups'][0]['lr']))

model_dir = 'training_results_supervised'
os.makedirs(model_dir, exist_ok=True)
final_model_path = os.path.join(model_dir, 'DeepONet.pth')
torch.save(model.state_dict(), final_model_path)

print("Training complete.")

k_list = []
khat_list = []

m = 50

state_dict = torch.load(final_model_path, weights_only=True)
model.load_state_dict(state_dict, strict=True)

lamarr = [1.0, 1.2]
lamarr1 = [1.6, 1.4]

for i in range(2):
    a_vals = np.zeros(101)
    lam = np.zeros(101)
    for j in range(101):
        a_vals[j] = lamarr[i] * (0.40 + 1.50 * spatial[j] ** 2)
        lam[j] = 0.10 + lamarr[i] * 0.10 * spatial[j]

    a_prime = 2.0 * 1.50 * lamarr[i] * spatial
    lam_bar_prime = 0.10 * lamarr[i]
    Q = (0.5 * lam_bar_prime
         - 0.5 * (a_prime / a_vals) * lam
         - 0.25 * lam ** 2 / a_vals)
    k = solveIntegralFD(m, a_vals, Q, 5 * lamarr1[i])

    karr = []
    fun1 = []
    fun2 = []
    karr.append(k)
    fun1.append(a_vals)
    fun2.append(Q)

    x_Fun1 = np.array(fun1, dtype=np.float32)
    x_Fun2 = np.array(fun2, dtype=np.float32)

    grid_size = 2 * nx + 1

    x_fun1 = np.repeat(x_Fun1[:, :, np.newaxis], grid_size, axis=2)
    x_fun2 = np.repeat(x_Fun2[:, :, np.newaxis], grid_size, axis=2)

    c_val = np.array([5 * lamarr1[i]], dtype=np.float32)
    AA = np.ones((grid_size, grid_size), dtype=np.float32)
    x_c_val = c_val[:, np.newaxis, np.newaxis] * AA

    x_fun1 = np.array(x_fun1, dtype=np.float32)
    x_fun2 = np.array(x_fun2, dtype=np.float32)
    x_c_val = np.array(x_c_val, dtype=np.float32)

    x_fun1 = x_fun1.reshape(x_fun1.shape[0], -1)
    x_fun2 = x_fun2.reshape(x_fun2.shape[0], -1)
    x_c_val = x_c_val.reshape(x_c_val.shape[0], -1)

    x_eval = np.stack((x_fun1, x_fun2, x_c_val), axis=1)
    x_eval = torch.from_numpy(x_eval).cuda()

    khat = model((x_eval, grid))
    khat = khat.cpu().detach().numpy().reshape(grid_size, grid_size)
    print(f"Sample {i} - khat shape: {khat.shape}")
    print(f"Sample {i} - k shape: {k.shape}")

    k_list.append(k)
    khat_list.append(khat)

print("===== Difference statistics: k vs k_hat =====")
for i in range(2):
    diff = k_list[i] - khat_list[i]
    diff_max = np.nanmax(diff)
    diff_min = np.nanmin(diff)
    diff_mean = np.nanmean(diff)
    diff_std = np.nanstd(diff)
    diff_absmax = np.nanmax(np.abs(diff))
    rel = diff_absmax / max(np.nanmax(np.abs(k_list[i])), 1e-12)
    print(f"\nSample {i+1} (lam = {lamarr[i]}):")
    print(f"  Max difference: {diff_max:.6f}")
    print(f"  Min difference: {diff_min:.6f}")
    print(f"  Mean difference: {diff_mean:.6f}")
    print(f"  Difference std: {diff_std:.6f}")
    print(f"  max|difference|: {diff_absmax:.6f}   relative error: {rel*100:.2f}%")


def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    golden_ratio = (5 ** .5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


tex_fonts = {
    "text.usetex": True,
    "font.family": "times",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(tex_fonts)

res = 5
fig = plt.figure(figsize=set_size(516, 0.99, (3, 2), height_add=1))
subfigs = fig.subfigures(nrows=3, ncols=1, hspace=0)

meshx, mesht = np.meshgrid(spatial, spatial, indexing='ij')

mask = np.abs(mesht) <= np.abs(meshx)

k0_masked = np.where(mask, k_list[0], np.nan)
k1_masked = np.where(mask, k_list[1], np.nan)
khat0_masked = np.where(mask, khat_list[0], np.nan)
khat1_masked = np.where(mask, khat_list[1], np.nan)
error0_masked = np.where(mask, khat_list[0] - k_list[0], np.nan)
error1_masked = np.where(mask, khat_list[1] - k_list[1], np.nan)

subfig = subfigs[0]
subfig.subplots_adjust(left=0.06, bottom=0, right=1, top=1.1, wspace=0.1, hspace=0)
subfig.suptitle(r"k(x, y) for $\gamma$ = 1.0, 1.2")

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))

ax[0].plot_surface(meshx, mesht, k0_masked, edgecolor="black", lw=0.2, rstride=res,
                   cstride=res, alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10, 15)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_zlabel(r"$k(x, y)$")
ax[0].set_xticks([-0.5, 0, 0.5])
ax[0].set_yticks([-0.5, 0, 0.5])

ax[1].plot_surface(meshx, mesht, k1_masked, edgecolor="black", lw=0.2, rstride=res,
                   cstride=res, alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_zlabel(r"$k(x, y)$")
ax[1].set_xticks([-0.5, 0, 0.5])
ax[1].set_yticks([-0.5, 0, 0.5])

ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

subfig = subfigs[1]
subfig.subplots_adjust(left=0.06, bottom=0, right=1, top=1, wspace=0.1, hspace=0)
subfig.suptitle(r"$\hat{k}(x, y)$ for $\gamma$ = 1.0, 1.2")

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))

ax[0].plot_surface(meshx, mesht, khat0_masked, edgecolor="black", lw=0.2, rstride=res,
                   cstride=res, alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10, 15)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_zlabel(r"$\hat{k}(x, y)$")
ax[0].set_xticks([-0.5, 0, 0.5])
ax[0].set_yticks([-0.5, 0, 0.5])

ax[1].plot_surface(meshx, mesht, khat1_masked, edgecolor="black", lw=0.2, rstride=res,
                   cstride=res, alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_zlabel(r"$\hat{k}(x, y)$")
ax[1].set_xticks([-0.5, 0, 0.5])
ax[1].set_yticks([-0.5, 0, 0.5])

ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

subfig = subfigs[2]
subfig.subplots_adjust(left=0.06, bottom=0, right=1, top=1, wspace=0.1, hspace=0)
subfig.suptitle(r"kernel error $k(x, y) - \hat{k}(x, y)$")

ax = subfig.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d", "computed_zorder": False})
for axis in [ax[0].xaxis, ax[0].yaxis, ax[0].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))
for axis in [ax[1].xaxis, ax[1].yaxis, ax[1].zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((0, 0, 0, 1))

ax[0].plot_surface(meshx, mesht, error0_masked, edgecolor="black", lw=0.2,
                   rstride=res, cstride=res, alpha=1, color="white", shade=False, rasterized=True)
ax[0].view_init(10, 15)
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_zlabel(r"$\hat{k}(x,y) - k(x, y)$", rotation=90)
ax[0].set_xticks([-0.5, 0, 0.5])
ax[0].set_yticks([-0.5, 0, 0.5])

ax[1].plot_surface(meshx, mesht, error1_masked, edgecolor="black", lw=0.2,
                   rstride=res, cstride=res, alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
ax[1].view_init(10, 15)
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_zlabel(r"$\hat{k}(x,y) - k(x, y)$", rotation=90)
ax[1].set_xticks([-0.5, 0, 0.5])
ax[1].set_yticks([-0.5, 0, 0.5])

ax[0].zaxis.set_rotate_label(False)
ax[1].zaxis.set_rotate_label(False)
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

fig.savefig('kernel_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Figure saved as kernel_comparison.png (dpi=600)")
