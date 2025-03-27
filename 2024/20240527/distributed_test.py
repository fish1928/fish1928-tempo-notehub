import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = torch.cuda.device_count()

def _reduce_scatter_along_first_dim(input_, device='cuda'):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = WORLD_SIZE

    dim_size = list(input_.size())
    assert (dim_size[0] % world_size == 0), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=device)
    
    torch.distributed._reduce_scatter_base(output, input_.contiguous())
    return output
# end

def _reduce_scatter_along_last_dim(input_, device='cuda'):
    """Reduce-scatter tensors on the last dimension."""
    world_size = WORLD_SIZE

    target_shape = list(input_.size())

    target_shape[-1] = target_shape[-1] // world_size

    input_ = input_.reshape(-1, input_.shape[-1])

    split_tensors = torch.split(input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1)

    concat_tensor = torch.cat(split_tensors, dim=0)

    output = _reduce_scatter_along_first_dim(concat_tensor, device).reshape(target_shape)

    return output
# end



""" All-Reduce example."""
def run(rank, size):
    device = torch.device(f'cuda:{rank}')
    
    tensor_in = torch.arange(6, dtype=torch.int64, device=device).reshape(2,3)
    # tensor_out = _reduce_scatter_along_first_dim(tensor_in, device=device)
    tensor_out = _reduce_scatter_along_last_dim(tensor_in, device=device)
    print(tensor_out)
# end




def init_process(rank, size, fn, backend='NCCL'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '3333' 
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
# end


if __name__ == "__main__":
    size = WORLD_SIZE
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    # end

    for p in processes:
        p.join()
    # end

    print('done')
# end
