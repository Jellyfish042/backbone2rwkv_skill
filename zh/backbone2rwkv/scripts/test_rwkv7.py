import os
import torch
import torch.nn as nn
from types import SimpleNamespace

os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_FLOAT_MODE"] = "bf16"

from rwkv7_cuda_kernel import load_wkv7_cuda_kernel, RUN_CUDA_RWKV7g
from rwkv7_time_mix import RWKV_Tmix_x070
from rwkv7_channel_mix import RWKV_CMix_x070


def test_rwkv7_modules():
    print("=" * 60)
    print("RWKV-7 Module Verification")
    print("=" * 60)

    args = SimpleNamespace(
        n_embd=128,
        dim_att=128,
        n_layer=2,
        head_size_a=64,
        head_size_divisor=8,
        my_testing="x070",
    )

    B = 1
    T = 37
    C = args.n_embd

    device = "cuda"
    dtype = torch.bfloat16

    print(f"\nConfig: B={B}, T={T}, C={C}, dtype={dtype}")
    print(f"Device: {device}")

    print("\n" + "-" * 60)
    print("Step 1: Loading CUDA kernel...")
    try:
        load_wkv7_cuda_kernel(head_size=64, use_training_kernel=True)
        print("CUDA kernel loaded successfully!")
    except Exception as e:
        print(f"Failed to load CUDA kernel: {e}")
        return False

    print("\n" + "-" * 60)
    print("Step 2: Creating modules...")

    tmix = RWKV_Tmix_x070(args, layer_id=0).to(device=device, dtype=dtype)
    cmix = RWKV_CMix_x070(args, layer_id=0).to(device=device, dtype=dtype)

    print(f"Time-Mix parameters: {sum(p.numel() for p in tmix.parameters()):,}")
    print(f"Channel-Mix parameters: {sum(p.numel() for p in cmix.parameters()):,}")

    print("\n" + "-" * 60)
    print("Step 3: Testing forward pass...")

    x = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)
    v_first = torch.empty_like(x)

    try:
        out_tmix, v_first = tmix(x, v_first)
        print(f"Time-Mix output shape: {out_tmix.shape}")
        print(f"Time-Mix output mean: {out_tmix.float().mean().item():.6f}")
        print("Time-Mix forward: OK")
    except Exception as e:
        print(f"Time-Mix forward FAILED: {e}")
        return False

    try:
        out_cmix = cmix(x)
        print(f"Channel-Mix output shape: {out_cmix.shape}")
        print(f"Channel-Mix output mean: {out_cmix.float().mean().item():.6f}")
        print("Channel-Mix forward: OK")
    except Exception as e:
        print(f"Channel-Mix forward FAILED: {e}")
        return False

    print("\n" + "-" * 60)
    print("Step 4: Testing backward pass...")

    try:
        loss_tmix = out_tmix.sum()
        loss_tmix.backward(retain_graph=True)

        grad_count = sum(1 for p in tmix.parameters() if p.grad is not None)
        total_params = sum(1 for p in tmix.parameters())
        print(f"Time-Mix gradients computed: {grad_count}/{total_params}")

        has_nan = any(torch.isnan(p.grad).any() for p in tmix.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in tmix.parameters() if p.grad is not None)

        if has_nan:
            print("Time-Mix backward: WARNING - NaN in gradients")
        elif has_inf:
            print("Time-Mix backward: WARNING - Inf in gradients")
        else:
            print("Time-Mix backward: OK")
    except Exception as e:
        print(f"Time-Mix backward FAILED: {e}")
        return False

    try:
        x2 = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)
        out_cmix2 = cmix(x2)
        loss_cmix = out_cmix2.sum()
        loss_cmix.backward()

        grad_count = sum(1 for p in cmix.parameters() if p.grad is not None)
        total_params = sum(1 for p in cmix.parameters())
        print(f"Channel-Mix gradients computed: {grad_count}/{total_params}")

        has_nan = any(torch.isnan(p.grad).any() for p in cmix.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in cmix.parameters() if p.grad is not None)

        if has_nan:
            print("Channel-Mix backward: WARNING - NaN in gradients")
        elif has_inf:
            print("Channel-Mix backward: WARNING - Inf in gradients")
        else:
            print("Channel-Mix backward: OK")
    except Exception as e:
        print(f"Channel-Mix backward FAILED: {e}")
        return False

    print("\n" + "-" * 60)
    print("Step 5: Testing multi-layer stack...")

    try:
        n_layers = 3
        tmix_layers = nn.ModuleList([RWKV_Tmix_x070(args, layer_id=i).to(device=device, dtype=dtype) for i in range(n_layers)])
        cmix_layers = nn.ModuleList([RWKV_CMix_x070(args, layer_id=i).to(device=device, dtype=dtype) for i in range(n_layers)])

        x = torch.randn(B, T, C, device=device, dtype=dtype, requires_grad=True)
        v_first = torch.empty_like(x)

        for i in range(n_layers):
            x_att, v_first = tmix_layers[i](x, v_first)
            x = x + x_att
            x = x + cmix_layers[i](x)

        loss = x.sum()
        loss.backward()

        print(f"Multi-layer forward-backward: OK")
        print(f"Final output shape: {x.shape}")
    except Exception as e:
        print(f"Multi-layer test FAILED: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


def test_fitting():
    print("=" * 60)
    print("RWKV-7 Fitting Test")
    print("=" * 60)

    args = SimpleNamespace(
        n_embd=128,
        dim_att=128,
        n_layer=2,
        head_size_a=64,
        head_size_divisor=8,
        my_testing="x070",
    )

    device = "cuda"
    dtype = torch.bfloat16
    B, T, C = 4, 23, args.n_embd

    load_wkv7_cuda_kernel(head_size=64, use_training_kernel=True)

    class SimpleRWKV(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.tmix = RWKV_Tmix_x070(args, layer_id=0)
            self.cmix = RWKV_CMix_x070(args, layer_id=0)
            self.head = nn.Linear(args.n_embd, args.n_embd, bias=False)

        def forward(self, x):
            v_first = torch.empty_like(x)
            x_att, v_first = self.tmix(x, v_first)
            x = x + x_att
            x = x + self.cmix(x)
            return self.head(x)

    model = SimpleRWKV(args).to(device=device, dtype=dtype)

    torch.manual_seed(42)
    inputs = torch.randn(B, T, C, device=device, dtype=dtype)
    targets = inputs * 0.5 + 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\nConfig: B={B}, T={T}, C={C}")
    print(f"Training for 100 steps...")
    print("-" * 60)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        output = model(inputs)
        loss = nn.functional.mse_loss(output.float(), targets.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == 99:
            print(f"Step {step:3d}: loss = {loss.item():.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]

    print("-" * 60)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")

    if final_loss < initial_loss * 0.01:
        print("\nFitting test: PASSED (loss reduced by >99%)")
        return True
    else:
        print("\nFitting test: FAILED (loss did not reduce enough)")
        return False


if __name__ == "__main__":
    success = test_rwkv7_modules()
    if success:
        success = test_fitting()
    exit(0 if success else 1)
