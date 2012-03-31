#!/usr/bin/env python
'''This script takes a contination file and a solution file and plots
the states next to a contination diagram. All frames are written to PNG files
which can later be concatenated into a movie.'''
# ==============================================================================
import os.path
import numpy as np
import paraview.simple as pv

import matplotlib.pyplot as pp
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib import rc
rc( 'font', **{'family':'sans-serif','sans-serif':['Helvetica']} )
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']))
rc('text', usetex=True)
# ==============================================================================
def _main():
    # Get file names from command line.
    args = _parse_arguments()

    #_plot_continuation()
    _state2png(args)
    return
# ==============================================================================
def _set_camera():
    view = pv.GetRenderView()
    #view = pv.GetActiveView()

    # set the camera
    #view.CameraViewUp = [0, 0, 1]
    #view.CameraViewAngle = 90
    view.CameraPosition = [0, 0, 1]
    view.CameraFocalPoint = [0, 0, 0]
    view.ViewSize = [600, 600]

    # set the time step
    view.ViewTime = 0

    # adjust camera angles
    #view.CameraParallelScale = 19.263253561273864
    #view.CameraViewUp = [-0.47908599379314365, 0.34724233720921105, 0.8061633642139294]
    #view.CameraPosition = [61.80685072696096, -15.494253739566437, 42.006468997427945]
    #view.CameraPosition = [0.0, 0.0, 20.0]
    #view.CameraClippingRange = [42.31038840424432, 115.0358420027582]
    #view.CameraFocalPoint = [2.3523564338684064, 2.2592620579362562e-15, -8.625427128948673e-16]
    # turn off axes visibility
    view.CenterAxesVisibility = 0
    view.OrientationAxesVisibility = 0
    view.UseOffscreenRenderingForScreenshots = 0

    return
# ==============================================================================
def _state2png(args):

    #pv._DisableFirstRenderCameraReset()

    # Open the data file.
    solution_e = pv.OpenDataFile( args.filename )
    #solution_e.FileRange = [0, 0]
    #solution_e.FilePrefix = args.solution_file
    #solution_e.XMLFileName = ''
    #solution_e.FilePattern = '%s'
    #solution_e.PointVariables = ['psi_']
    #solution_e.ElementBlocks = ['block_5']

    #AnimationScene1 = GetAnimationScene()
    #AnimationScene1.EndTime = 5040.0
    #AnimationScene1.PlayMode = 'Snap To TimeSteps'

    _set_camera()

    filenames = {}

    # create arrays
    array_name = '|psi|^2'
    calc1 = pv.Calculator( ResultArrayName = array_name )
    calc1.AttributeMode = 'point_data'
    calc1.Function = 'psi__X^2 + psi__Y^2'
    filenames[array_name] = 'abs.png'

    # Use Calculator conditionals <http://www.itk.org/Wiki/ParaView/Users_Guide/Calculator>
    # to imitate the atan2() function with atan().
    # See <http://en.wikipedia.org/wiki/Atan2#Definition>.
    array_name = 'arg(psi)'
    calc2 = pv.Calculator( ResultArrayName = array_name )
    calc2.AttributeMode = 'point_data'
    calc2.Function = 'if( psi__X>0, atan(psi__Y/psi__X),' + \
                     'if( psi__X<0,' + \
                         'if ( psi__Y<0, - %s + atan(psi__Y/psi__X), %s + atan(psi__Y/psi__X) ),' % ( np.pi, np.pi ) + \
                     'if ( psi__Y>0, %s, if( psi__Y<0, -%s, 0.0 ) )' % ( np.pi/2, np.pi/2 ) + \
                     '))'
    filenames[array_name] = 'arg.png'

    _absarg2png( filenames )

    #for filename in filenames.items():
        #_autocrop( filename )

    return
# ==============================================================================
def _plot_continuation():
    '''Main function.'''

    pv._DisableFirstRenderCameraReset()

    solution_e = pv.OpenDataFile( args.solution_file )

    #AnimationScene1 = GetAnimationScene()
    solution_e.FileRange = [0, 0]
    solution_e.FilePrefix = solution_file
    solution_e.XMLFileName = ''
    solution_e.FilePattern = '%s'

    #AnimationScene1.EndTime = 5040.0
    #AnimationScene1.PlayMode = 'Snap To TimeSteps'

    solution_e.PointVariables = ['psi_']
    solution_e.ElementBlocks = ['block_5']

    # create calculator filter that computes the Cooper pair density
    calc1 = pv.Calculator( ResultArrayName = '|psi|^2' )
    calc1.AttributeMode = 'point_data'
    calc1.Function = 'psi__X^2 + psi__Y^2'

    # Use Calculator conditionals <http://www.itk.org/Wiki/ParaView/Users_Guide/Calculator>
    # to imitate the atan2() function with atan().
    # See <http://en.wikipedia.org/wiki/Atan2#Definition>.
    calc2 = pv.Calculator( ResultArrayName = 'arg(psi)' )
    calc2.AttributeMode = 'point_data'
    calc2.Function = 'if( psi__X>0, atan(psi__Y/psi__X),' + \
                     'if( psi__X<0,' + \
                         'if ( psi__Y<0, - %s + atan(psi__Y/psi__X), %s + atan(psi__Y/psi__X) ),' % ( np.pi, np.pi ) + \
                     'if ( psi__Y>0, %s, if( psi__Y<0, -%s, 0.0 ) )' % ( np.pi/2, np.pi/2 ) + \
                     '))'

    # set view angle etc.
    _set_camera()

    data_representation = pv.Show()

    # read the continuation file
    continuation_data = np.loadtxt( continuation_file )

    plot_columns = {'mu': 1, 'energy': 9}

    # plot the contination data
    x_values = continuation_data[ 0:, plot_columns['mu'] ]
    y_values = continuation_data[ 0:, plot_columns['energy'] ]

    # prepare the plot
    fig = pp.figure()
    # Adjust the size of the figure to host two figures next to each other
    # with little distortion.
    default_size = fig.get_size_inches()
    fig.set_size_inches( ( default_size[0]*2, default_size[1] ),
                         forward = True
                       )

    # draw first diagram
    diagram_ax = fig.add_subplot( 1, 3, 3 )
    pp.plot( x_values, y_values, 'r-' )
    max_val = int(10*max( y_values )) / 10.0 # round to -0.3, -0.4, ...
    pp.ylim( -1.0, max_val )
    pp.xlabel( '$\mu$' )
    pp.ylabel( '$F/F_0$' )

    # prepare for the blue moving dot
    line, = diagram_ax.plot( [], [], 'bo' )

    ## take screenshot
    #filename = 'test0.png'
    #tstep = 0
    
    #_take_screenshot( view, tstep, data_representation, filename )
    # plot screenshot
    fig.add_subplot( 1, 3, 1 )
    # turn off ticks
    pp.title('$|\psi|^2$')
    pp.xticks( [] )
    pp.yticks( [] )
    pp.box( 'off' )

    fig.add_subplot( 1, 3, 2 )
    pp.title('$\mathrm{arg}\,\psi$')
    # turn off ticks
    pp.xticks( [] )
    pp.yticks( [] )
    pp.box( 'off' )

    # Creating a movie won't work before a proper PNG filter is in ffmpeg/libav,
    # see
    # http://stackoverflow.com/questions/4092927/generating-movie-from-python-without-saving-individual-frames-to-files
    #outf = 'test.avi'
    #rate = 1
    #cmdstring = ('/usr/bin/ffmpeg',
                #'-r', '%d' % rate,
                #'-f','image2pipe',
                #'-vcodec', 'png',
                #'-i', 'pipe:', outf
                #)
    #p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

    # Create cache folder for all the PNGs.
    cache_folder = os.path.join( folder, 'data' )
    if not os.path.exists( cache_folder ):
        os.makedirs( cache_folder )

    # begin the loop
    num_steps = len( solution_e.TimestepValues )
    max_number_of_digits = int( np.ceil( np.log10( num_steps + 1 ) ) )
    for k, tstep in enumerate( solution_e.TimestepValues ):

        # plot the blue moving dot
        line.set_data( continuation_data[ k, plot_columns['mu'] ],
                       continuation_data[ k, plot_columns['energy'] ]
                     )

        # Take screenshot.
        # For int formatting see
        # http://stackoverflow.com/questions/733454/best-way-to-format-integer-as-string-with-leading-zeros
        suffix = '%0*d.png' % ( max_number_of_digits, int(tstep) )
        filenames = { 'abs': os.path.join( cache_folder, 'abs%s' % suffix ),
                      'arg': os.path.join( cache_folder, 'arg%s' % suffix )
                    }

        view.ViewTime = tstep
        _take_screenshots( data_representation, filenames )

        # autocrop image
        _autocrop( filenames['abs'] )
        _autocrop( filenames['arg'] )

        # Plot abs.
        fig.add_subplot( 1, 3, 1 )
        pp.imshow( mpimg.imread( filenames['abs'] ),
                   cmap = cm.gray,
                   vmin = 0.0,
                   vmax = 1.0
                 )
        # set colorbar
        if k == 0:
            pp.colorbar( ticks = [0, 0.5, 1],
                         shrink = 0.6
                       )

        # Plot arg.
        fig.add_subplot( 1, 3, 2 )
        pp.imshow( mpimg.imread( filenames['arg'] ),
                   cmap = cm.hsv,
                   vmin   = -np.pi,
                   vmax   = np.pi
                 )
        # set colorbar
        if k == 0:
            cbar1 = pp.colorbar( ticks=[-np.pi, 0, np.pi],
                                 shrink = 0.6
                              )
            cbar1.set_ticklabels( ['$-\pi$', '$0$', '$\pi$'] )

        # draw the thing
        #pp.draw()
        #pp.show()
        patch_file = os.path.join( cache_folder, 'patched%s' % suffix )
        pp.savefig(patch_file, format='png')
        #pp.savefig( p.stdin, format='jpg' )

        print '%d / %d' % (k, num_steps)
        
    return
# ==============================================================================
def _parse_arguments():
    '''Python 2.7 argument parser.'''
    import argparse

    parser = argparse.ArgumentParser( description='Prettyprint states and energy.' )

    parser.add_argument( 'filename',
                         metavar = 'FILE_OR_FOLDER',
                         type     = str,
                         help     = 'single file or directory containing solution.e and continuationData.dat'
                       )

    parser.add_argument( '-d', '--data',
                         dest     = 'plot_data',
                         #required = True,
                         action = 'store_true',
                         default = False,
                         help     = 'plot the data (default: false, just plot the grid)'
                       )

    return parser.parse_args()

    #directory         = args.solution_directory
    #solution_file     = os.path.join( directory, 'solution.e' )
    #continuation_file = os.path.join( directory, 'continuationData.dat' )
    #return directory, solution_file, continuation_file
# ==============================================================================
def _absarg2png( filenames ):
    '''Create images of |psi|^2 and arg(psi).
    The 'filenames' arguments is a dictionary with
    key:value = array_name:file_name.'''

    data_representation = pv.Show()
    # Reset the camera here to get the whole object.
    view = pv.GetRenderView()
    view.ResetCamera()

    # create calculator filter that computes the Cooper pair density
    array_name = filenames.keys()[1]
    # make background green
    view.Background =  [0.0, 1.0, 0.0]
    #data_representation.ScalarOpacityFunction = pv.CreatePiecewiseFunction()
    data_representation.ColorArrayName = array_name
    data_representation.LookupTable = \
        pv.GetLookupTableForArray(array_name, 1,
                                  RGBPoints  = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                                  LockScalarRange = 1
                                  )
    pv.WriteImage( filenames.values()[1] )
    #pv.Render()

    array_name = filenames.keys()[0]
    # make backgroun gray
    view.Background =  [0.5, 0.5, 0.5]
    #data_representation.ScalarOpacityFunction = pv.CreatePiecewiseFunction()
    data_representation.ColorArrayName = array_name
    data_representation.LookupTable = \
        pv.GetLookupTableForArray(array_name, 1,
                                  RGBPoints =  _create_circular_hsv_colormap(),
                                  LockScalarRange = 1
                                  )
    # Don't interpolate scalars as otherwise, the circular HSV color map
    # gets messed up at the pi/-pi seam.
    data_representation.InterpolateScalarsBeforeMapping = 0
    pv.WriteImage( filenames.values()[0] )

    return
# ==============================================================================
def _create_circular_hsv_colormap():
    '''Returns the notorious red-blue-red circular color map
    in ParaView format.'''
    import matplotlib

    n = 256 # number of colors
    hsv = np.empty( [n, 1, 3], dtype=float )

    # generate the hsv space
    x = np.linspace( 0, 1, num=n, endpoint=True, retstep=False )
    for k in range(n):
        hsv[k, 0, : ] = [ x[k], 1, 1 ]

    # transform to rgb
    rgb = matplotlib.colors.hsv_to_rgb( hsv )

    # cast this into a ParaView-friendly format
    lower = -np.pi
    upper =  np.pi
    cmap = np.empty( 4*n, dtype=float )
    for k in range(n):
        cmap[ 4*k:4*(k+1) ] = [ lower + (upper-lower) * k/(n-1),
                                rgb[k][0][0], rgb[k][0][1], rgb[k][0][2] ]

    return cmap
# ==============================================================================
def _autocrop( filename ):
    '''Autocrop an image stored in file.'''
    import PythonMagick

    image = PythonMagick.Image( filename )
    image.trim()
    image.write( filename )

    return
# ==============================================================================
if __name__ == '__main__':
    _main()
# ==============================================================================
