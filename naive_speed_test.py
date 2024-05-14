import torch
dim = 4096*2
A = torch.randn((dim,dim)).to("cuda")
B = torch.randn((dim,dim)).to("cuda")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
num_iters = 100
start.record()
for i in range(num_iters):
    C = A @ B
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end)/num_iters)
