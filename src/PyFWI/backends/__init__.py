"""Compute backends of PyFWI: what runs the kernels (roadmap Phase 2).

The propagation code never calls a GPU library directly: it asks a *backend* to allocate buffers,
copy data, compile kernels and launch them. ``opencl.OpenCLBackend`` wraps PyOpenCL, the engine of
PyFWI 0.1.10; a CUDA backend comes with step 2.3, and the tests use a fake one written with NumPy.

Every backend offers the same small interface:

==========================================  ===============================================================
``allocate(host, margin_bytes, fill)``      device buffer holding ``host``, inside an allocation
                                            ``[margin | data | margin]`` whose margins hold ``fill``;
                                            returns ``(buffer, allocation)``
``allocate_empty(nbytes)``                  device buffer without margins, content undefined
``nbytes(buffer)``                          size of a buffer in bytes
``alignment``                               bytes to which a buffer inside an allocation must be aligned
``to_device(buffer, host, blocking=True)``  copy host -> device; returns an event
``to_host(host, buffer, blocking=True)``    copy device -> host; returns an event
``copy(target, source, nbytes, offset)``    copy device -> device, from byte ``offset`` of ``source``
``wait(events)``, ``finish()``              wait for some events, or for everything enqueued
``build(source)``                           compile a program from its source (``#define`` block included)
``kernel(program, name)``                   a kernel of a program, retrieved once and reused
``set_args(kernel, *args)``                 set every argument of a kernel
``set_arg(kernel, index, value)``           set one argument
``launch(kernel, global_size)``             run a kernel on a grid, with the arguments set before
``call(kernel, global_size, *args)``        set every argument and run
``capture(record)`` (optional)              record the launches of ``record()`` into a graph that
                                            ``launch()`` replays (CUDA only, step 3.7)
==========================================  ===============================================================

A non-blocking copy keeps using the host array until its event completes: the caller keeps the
array alive and unchanged until then (``wait`` or ``finish``).
"""
