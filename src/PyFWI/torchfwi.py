import matplotlib.pyplot as plt
import numpy as np

import PyFWI.processing as process
import torch


# Step 4.5 of the fork: tensors on the GPU. PyFWI's propagators take and return NumPy arrays, so a CUDA
# tensor goes to the host and back; the exchanges with CuPy go through DLPack, which shares the memory
# instead of copying it (a tensor and the CuPy array made from it have the same data pointer).

def _to_numpy(tensor):
    """The values of ``tensor`` as a float32 NumPy array: a CUDA tensor through CuPy (DLPack, no copy on
    the GPU), then one copy to the host."""
    tensor = tensor.detach()
    if tensor.is_cuda:
        import cupy
        return cupy.from_dlpack(tensor).get().astype(np.float32)
    return tensor.cpu().numpy().astype(np.float32)


def _like(values, reference):
    """``values`` (NumPy) as a tensor on the device of ``reference``: for a CUDA tensor, one copy to the GPU
    into a CuPy array, which the tensor then shares (DLPack)."""
    if reference.is_cuda:
        import cupy
        device = reference.device.index or 0
        if device == cupy.cuda.runtime.getDevice():
            return torch.from_dlpack(cupy.asarray(values))
        with cupy.cuda.Device(device):  # another GPU than the current one
            return torch.from_dlpack(cupy.asarray(values))
    return torch.tensor(values)

class Fwi(torch.autograd.Function):
    """
        Fwi is a custom function to integrate PyFWI wo PyTorch.
        The gradient of cost function with respect to model parameters
        are obtained using adjoint state method.
        
        Parameters
        ----------
        W : WavePropagator
            A class of WavePropagator to perform the modeling and backpropagations
        vp : Tensor
            P-wave velocity
        vs : Tensor
            S-wave velocity
        rho : Tensor
            Density

        Returns
        -------
        taux : Tensor
            Normal stress in x-direction 
        tauz : Tensor
            Normal stress in z-direction 
        """
    @staticmethod
    def forward(ctx, W, vp, vs, rho):
        
        # CPU or CUDA tensors (step 4.5 of the fork: the outputs are on the device of vp).
        model = {'vp': _to_numpy(vp),
                 'vs': _to_numpy(vs),
                 'rho': _to_numpy(rho)
                 }

        # Call the forward modelling
        db_obs = W.forward_modeling(model, show=False)  # show=True can show the propagation of the wave

        # ctx.save_for_backward(vp)        
        ctx.WavePropagator = W
        ctx.model_tensor = vp  # the device of the gradient
                
        return _like(db_obs['taux'], vp),\
            _like(db_obs['tauz'], vp)

    
    @staticmethod
    def backward(ctx, adj_tx, adj_tz):
        """
        backward calculates the gradient of cost function with
        respect to model parameters

        Parameters
        ----------
        adj_tx : Tensor
            Adjoint of recoreded normal stress in x-direction
        adj_tz : Tensor
            Adjoint of recoreded normal stress in z-direction

        Returns
        -------
        Tensor
            Gradient of cost function w.r.t. model parameters
        """
        adj_taux = _to_numpy(adj_tx)
        adj_tauz = _to_numpy(adj_tz)
        
        # If shape is (nt, nr, ns) switch it to (nt, nr * ns)
        if adj_tauz.ndim == 3:
            (nt, nr, ns) = adj_taux.shape
            adj_taux = np.reshape(adj_taux, (nt, nr * ns), 'F')
            adj_tauz = np.reshape(adj_tauz, (nt, nr * ns), 'F')
            
        adj_src = process.prepare_residual(
            {
                # 'vx':adj_vx.detach().numpy(),
                # 'vz': adj_vz.detach().numpy(),
                'taux': adj_taux,
                'tauz': adj_tauz,
                # 'tauxz': adj_txz.detach().numpy(),
             }, 1)
                
        result = ctx.WavePropagator.gradient(adj_src, False)
        
        return (None, 
                _like(result['vp'], ctx.model_tensor), 
                _like(result['vs'], ctx.model_tensor), 
                _like(result['rho'], ctx.model_tensor)
                )
    