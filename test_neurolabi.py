import numpy

from cffi import FFI
ffi = FFI()
ffi.cdef("""
typedef unsigned char uint8;
typedef struct
  { int      kind;
    int      width;
    int      height;
    int      depth;
    char    *text;
    uint8   *array;   // array of pixel values lexicographically ordered on (z,y,x,c)
  } Stack;

typedef struct _Stack_Watershed_Workspace {
  int *array;
  Stack *mask; /* 1~244 seed; 255 barrier */
  int conn;
  int min_level;
  int start_level;
  double *weights;
} Stack_Watershed_Workspace;

Stack *Make_Stack(int kind, int width, int height, int depth);
Stack_Watershed_Workspace* Make_Stack_Watershed_Workspace(const Stack *stack);
Stack* Stack_Watershed(const Stack *stack, Stack_Watershed_Workspace *ws);

""")

neurolabi = ffi.dlopen('/Users/bergs/Documents/workspace/NeuTu/neurolabi/c/lib/libneurolabi.dylib')

# Constants from image_lib.h
GREY      = 1
GREY16    = 2
COLOR     = 3
FLOAT32   = 4

def neurolabi_watershed_3d( volume_zyx, seeds_zyx, mask_zyx=None ):
    """
    Run the neurolabi watershed.
    
    NOTE: neurolabi auto-inverts the input data.  The input to this function should have seeds near the MAXIMA.
    """
    assert volume_zyx.dtype == numpy.uint8, "Watershed input must be uint8"
    assert seeds_zyx.dtype == numpy.uint8, "Watershed seeds must be uint8"
    
    assert volume_zyx.ndim == seeds_zyx.ndim == 3, "All inputs must be 3D"
    assert mask_zyx is None or (mask_zyx.ndim == 3), "All inputs must be 3D"
    assert volume_zyx.shape == seeds_zyx.shape, \
        "Shape mismatch: {} != {}".format( volume_zyx.shape, seeds_zyx.shape )
    assert volume_zyx.shape == seeds_zyx.shape, "Seeds shape must match input shape"
    assert mask_zyx is None or mask_zyx.shape == seeds_zyx.shape, "Mask shape must match seeds shape: {} != {}".format( mask_zyx.shape, seeds_zyx.shape )
    
    # Always copy. We're going to modify the data
    volume_zyx = numpy.array( volume_zyx )

    # Always copy. We're going to modify the seeds
    seeds_zyx = numpy.array( seeds_zyx )

    if mask_zyx is not None and not mask_zyx.flags['C_CONTIGUOUS']:
        mask_zyx = numpy.array( mask_zyx )

    # Copy mask into seeds
    if mask_zyx is not None:
        # In neurolabi's watershed, a seed of 255 means "mask"
        seeds_zyx[mask_zyx == 0] = 255

    depth, height, width = volume_zyx.shape
    vol_stack = neurolabi.Make_Stack(GREY, width, height, depth ) # xyz
    p_volume = ffi.cast("uint8 *", volume_zyx.ctypes.data)
    vol_stack.array = p_volume

    seeds_stack = neurolabi.Make_Stack(GREY, width, height, depth ) # xyz
    p_mask = ffi.cast("uint8 *", seeds_zyx.ctypes.data)
    seeds_stack.array = p_mask
    
    workspace = neurolabi.Make_Stack_Watershed_Workspace(vol_stack)
    workspace.conn = 26
    workspace.mask = seeds_stack # Both mask and seeds are provided in this field.
    ws_stack = neurolabi.Stack_Watershed(vol_stack, workspace)

    # Extract from buffer
    buf = ffi.buffer( ws_stack.array, numpy.prod(volume_zyx.shape) )
    ws_1d = numpy.frombuffer( buf, dtype=numpy.uint8 )
    ws_array = ws_1d.reshape( volume_zyx.shape )
    return ws_array

def vigra_watershed_3d( volume_zyx, seeds_zyx, mask_zyx=None ):
    if mask_zyx is None:
        # Fast path
        return vigra.analysis.watersheds( volume_zyx,
                                          seeds=seeds_zyx.astype(numpy.uint32),
                                          method='RegionGrowing')
    # Slow path:
    # Run non-turbo watershed, using a stopping condiction to exclude the mask
    volume_zyx = numpy.array( volume_zyx )
    volume_zyx[volume_zyx == 255] = 254
    volume_zyx[mask_zyx == 0] = 255
    
    seeds_zyx = numpy.array( seeds_zyx )
    seeds_zyx[mask_zyx == 0] = 0
    return vigra.analysis.watersheds( volume_zyx,
                                      seeds=seeds_zyx.astype(numpy.uint32),
                                      max_cost=253,
                                      terminate=vigra.analysis.SRGType.StopAtThreshold,
                                      method='RegionGrowing')

if __name__ == "__main__":
    
    import sys
    import vigra
    import h5py
    from pathHelpers import PathComponents
    from timer import Timer

    if len(sys.argv) < 2:
        sys.stdout.write("Usage: {} my_input.h5/some/volume\n".format(sys.argv[0]))
        sys.exit(1)
    
    pc = PathComponents(sys.argv[1])
    
    with h5py.File(pc.externalPath, 'r') as f:
        raw_zyx = f[pc.internalPath][:]
        if raw_zyx.ndim > 3:
            # Remove channel
            raw_zyx = raw_zyx[...,0]
        assert raw_zyx.ndim == 3

    inverted_zyx = 255 - raw_zyx
    binary_seeds = inverted_zyx <= 100
    labeled_seeds = vigra.analysis.labelVolumeWithBackground(binary_seeds.astype( numpy.uint8 ), background_value=0)
    labeled_seeds = labeled_seeds.astype( numpy.uint8 )
    print "Volume has {} seeds".format( labeled_seeds.max() )
    
    mask = numpy.zeros( raw_zyx.shape, dtype=numpy.uint8 )
    depth, height, width = raw_zyx.shape
    mask[:, :height/4, :] = 1
    
    with Timer() as timer:
        watershed_vol = neurolabi_watershed_3d(raw_zyx, labeled_seeds, mask)
        #watershed_vol = neurolabi_watershed_3d(raw_zyx, labeled_seeds, None)
    print "Neurolabi runtime: {}".format( timer.seconds() )

    with h5py.File('/tmp/watershed_output.h5', 'w') as f:
        f['watershed'] = watershed_vol

    with Timer() as timer:
        vigra_watershed, max_label = vigra_watershed_3d( inverted_zyx, labeled_seeds, mask )
        #vigra_watershed, max_label = vigra_watershed_3d( inverted_zyx, labeled_seeds, None )

    print "Vigra runtime: {}".format( timer.seconds() )
        
    print "DONE"

    # Show in volumina

    SHOW_GUI = False
    if SHOW_GUI:
        from PyQt4.QtGui import QApplication
        import volumina
        
        app = QApplication([])
        
        viewer = volumina.viewer.Viewer()
        viewer.show()
        viewer.addGrayscaleLayer( raw_zyx ).name = "raw"
        viewer.addGrayscaleLayer( inverted_zyx ).name = "inverted"
        viewer.addRandomColorsLayer( watershed_vol ).name = "neurolabi watershed"
        viewer.addRandomColorsLayer( vigra_watershed.view(numpy.ndarray) ).name = "vigra watershed"
        viewer.addRandomColorsLayer( labeled_seeds ).name = "seeds"
        
        app.exec_()
