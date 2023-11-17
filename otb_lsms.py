# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:22:54 2020

@author: disbr007
"""
import argparse
import datetime
import logging.config
import os
import re
import subprocess
from subprocess import PIPE
import sys

from tqdm import tqdm

from misc_utils.logging_utils import LOGGING_CONFIG, create_logger
# from misc_utils.RasterWrapper import Raster


#### Set up logger
handler_level = 'INFO'
logging.config.dictConfig(LOGGING_CONFIG(handler_level))
logger = logging.getLogger(__name__)


#### Function definition
def run_subprocess(command, debug_filters=None, return_lines=None):
    """Run the commmand passed as a subprocess.
    Optionally use debug_filter to route specific messages from the
    subprocess to the debug logging level.
    Optionally matching specific messages and returning them.

    Parameters
    ----------
    command : str
        The command to run.
    debug_filters : list
        List of strings, where if any of the strings are in
        the subprocess message, the message will be routed to
        debug.
    return_lines : list
        List of tuples of (string, int) where the strings are
        regex patterns and int are group number within match to
        return.

    Returns
    ------
    return_lines : list / None
    """
    message_values = []
    proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):  # replace '' with b'' for Python 3
        message = line.decode()
        if any([f in message.lower() for f in debug_filters]):
            logger.debug(message)
        else:
            logger.info(message)
        if return_lines:
            for pattern, group_num in return_lines:
                pat = re.compile(pattern)
                match = pat.search(message)
                if match:
                    value = match.group(group_num)
                    message_values.append(value)

    proc_err = ""
    for line in iter(proc.stderr.readline, b''):
        proc_err += line.decode()
    if proc_err:
        logger.info(proc_err)
    output, error = proc.communicate()
    logger.debug('Output: {}'.format(output.decode()))
    logger.debug('Err: {}'.format(error.decode()))

    return message_values


def otb_lsms(img, mode='vector',
             spatialr=5, ranger=15, minsize=50,
             tilesize_x=500, tilesize_y=500,
             out=None,
             ram=256,
             overwrite=False):
    """
    Run the Orfeo Toolbox LargeScaleMeanShift command via the command
    line. Requires that OTB environment is activated.

    Parameters
    ----------
    img : os.path.abspath
        Path to raster to be segmented.
    mode : str
        Format to write segmentation. One of 'raster' or 'vector'.
    spatialr : INT
        Spatial radius -- Default value: 5
        Radius of the spatial neighborhood for averaging.
        Higher values will result in more smoothing and higher processing time.
    ranger : FLOAT
        Range radius -- Default value: 15
        Threshold on spectral signature euclidean distance (expressed in radiometry unit)
        to consider neighborhood pixel for averaging.
        Higher values will be less edge-preserving (more similar to simple average in neighborhood),
        whereas lower values will result in less noise smoothing.
        Note that this parameter has no effect on processing time..
    minsize : INT
        Minimum Segment Size -- Default value: 50
        Minimum Segment Size. If, after the segmentation, a segment is of size strictly
        lower than this criterion, the segment is merged with the segment that has the
        closest sepctral signature.
    tilesize_x : INT
        Size of tiles in pixel (X-axis) -- Default value: 500
        Size of tiles along the X-axis for tile-wise processing.
    tilesize_y : INT
        Size of tiles in pixel (Y-axis) -- Default value: 500
        Size of tiles along the Y-axis for tile-wise processing.
    out : os.path.abspath
        Path to write segmentation to.

    Returns
    -------
    Path to out_vector.

    """
    # Log input image information
    # TODO: This is not working due to OTB using it's own GDAL.
    # src = Raster(img)
    # x_sz = src.x_sz
    # y_sz = src.y_sz
    # depth = src.depth
    # src = None

    # logger.info("""Running OTB Large-Scale-Mean-Shift...
                # Input image: {}
                # # Image X Size: {}
                # # Image Y Size: {}
                # # Image # Bands: {}
                # Spatial radius: {}
                # Range radius: {}
                # Min. segment size: {}
                # Tilesizex: {}
                # Tilesizey: {}
                # Output mode: {}
                # Output segmentation: {}""".format(img, x_sz, y_sz, depth,
                #                                   spatialr, ranger, minsize,
                #                                   tilesize_x, tilesize_y, mode, out))

    logger.info("""Running OTB Large-Scale-Mean-Shift...
                Input image: {}
                Spatial radius: {}
                Range radius: {}
                Min. segment size: {}
                Tilesizex: {}
                Tilesizey: {}
                Output mode: {}
                Output segmentation: {}""".format(img, spatialr, ranger, minsize,
                                                  tilesize_x, tilesize_y, mode, out).strip('\t'))
    # Check if output segmentation exists
    if os.path.exists(out):
        if not overwrite:
            logger.warning('Output segmentation exists. Exiting.')
            sys.exit()
        else:
            logger.warning('Output segmentation exists, will be overwritten.')

    # Build command
    cmd = """otbcli_LargeScaleMeanShift
             -in {}
             -spatialr {}
             -ranger {}
             -minsize {}
             -tilesizex {}
             -tilesizey {}
             -mode.{}.out {}""".format(img, spatialr, ranger, minsize,
                                       tilesize_x, tilesize_y, mode, out)
    # Remove whitespace, newlines
    cmd = cmd.replace('\n', '')
    cmd = ' '.join(cmd.split())
    logger.info(cmd)

    # Messages from OTB to filter to DEBUG
    debug_filters = ['estimat', ' will be written in ',
                     'unable to remove file']
    run_time_start = datetime.datetime.now()
    temp_files = run_subprocess(cmd, debug_filters=debug_filters,
                                return_lines=[('Unable to remove file (.*)', 1)])
    run_time_finish = datetime.datetime.now()
    run_time = run_time_finish - run_time_start
    too_fast = datetime.timedelta(seconds=5)
    if run_time < too_fast:
        logger.warning('Execution completed quickly, likely due to an error. '
                       'Did you activate OTB env first?\n'
                       'C:\OTB-7.1.0-Win64\OTB-7.1.0-Win64\otbenv.bat\nor\n'
                       'module load otb/6.6.1')

    logger.info('Large-Scale-Mean-Shift finished. Runtime: {}'.format(str(run_time)))

    if temp_files:
        logger.info('Removing unremoved temporary files...')
        for tf in tqdm(temp_files):
            tf_path = tf.strip()
            logger.debug('Removing: {}'.format(tf_path))
            try:
                os.remove(tf_path)
            except Exception as e:
                logger.error('Unable to remove temp file: {}'.format(tf_path))
                logger.error(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=r"Wrapper for Orfeo Toolbox LargeScaleMeanShift "
                    r"segmentation. OTB environment must be activated: "
                    r"C:\OTB-7.1.0-Win64\OTB-7.1.0-Win64\otbenv.bat")

    parser.add_argument('-i', '--image_source',
                        type=os.path.abspath,
                        help='Image to segment.')
    parser.add_argument('-o', '--out',
                        type=os.path.abspath,
                        help='Output filename. If ".shp" mode will be vector. If ".tif" mode will be raster.')
    parser.add_argument('-od', '--out_dir',
                        type=os.path.abspath,
                        help="""Alternatively to specifying out_vector path, specify
                                just the output directory and the name will be
                                created in a standardized fashion following:
                                [input_filename]_sr[sr]rr[rr]ms[ms]tx[tx]ty[ty].shp""")
    parser.add_argument('-m', '--mode',
                        type=str,
                        choices=['vector', 'raster'],
                        help='Output mode for labeled segmentation.')
    parser.add_argument('-sr', '--spatial_radius',
                        type=int,
                        default=5,
                        help="""Spatial radius -- Default value: 5
                                Radius of the spatial neighborhood for averaging.
                                Higher values will result in more smoothing and
                                higher processing time.""")
    parser.add_argument('-rr', '--range_radius',
                        type=float,
                        default=15,
                        help="""Range radius -- Default value: 15
                                Threshold on spectral signature euclidean distance
                                (expressed in radiometry unit) to consider neighborhood
                                pixel for averaging. Higher values will be less
                                edge-preserving (more similar to simple average in neighborhood),
                                whereas lower values will result in less noise smoothing.
                                Note that this parameter has no effect on processing time.""")
    parser.add_argument('-ms', '--minsize',
                        type=int,
                        default=50,
                        help="""Minimum Segment Size -- Default value: 50
                                Minimum Segment Size. If, after the segmentation, a segment is of
                                size strictly lower than this criterion, the segment is merged with
                                the segment that has the closest sepctral signature.""")
    parser.add_argument('-tx', '--tilesize_x',
                        type=int,
                        default=500,
                        help="""Size of tiles in pixel (X-axis) -- Default value: 500
                                Size of tiles along the X-axis for tile-wise processing.""")
    parser.add_argument('-ty', '--tilesize_y',
                        type=int,
                        default=500,
                        help="""Size of tiles in pixel (Y-axis) -- Default value: 500
                                Size of tiles along the Y-axis for tile-wise processing.""")
    parser.add_argument('-l', '--log_file',
                        type=os.path.abspath,
                        help='Path to write log file to.')
    parser.add_argument('-ld', '--log_dir',
                        type=os.path.abspath,
                        help="""Directory to write log to, with standardized name following
                                out vector naming convention.""")
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output segmentation if it exists. (Warning message'
                             'will appear)')

    args = parser.parse_args()

    image_source = args.image_source
    out = args.out
    out_dir = args.out_dir
    mode = args.mode
    spatialr = args.spatial_radius
    ranger = args.range_radius
    minsize = args.minsize
    tilesize_x = args.tilesize_x
    tilesize_y = args.tilesize_y
    overwrite = args.overwrite

    # Set up console logger
    handler_level = 'INFO'
    logger = create_logger(__name__, 'sh',
                           handler_level=handler_level)

    # Build out path and determine mode
    if out:
        mode_lut = {'.shp': 'vector', '.tif': 'raster'}
        ext = os.path.splittext(out)[1]
        if not mode:
            mode = mode_lut[ext]
        else:
            if mode != mode_lut[ext]:
                logger.error("""Selected mode does not match out file extension:
                                mode: {} != {}""".format(mode, ext))
                sys.exit()

    if out is None:
        if mode is None:
            logger.error("""Must supply one of "out" with extension or "mode" in order to
                            determine output type (raster|vector).""")
            sys.exit()
        else:
            ext_lut = {'vector': 'shp', 'raster': 'tif'}
            ext = ext_lut[mode]
        if out_dir is None:
            out_dir = os.path.dirname(image_source)
        out_name = os.path.basename(image_source).split('.')[0]
        out_name = '{}_sr{}_rr{}_ms{}_tx{}_ty{}.{}'.format(out_name, spatialr, str(ranger).replace('.', 'x'),
                                                           minsize, tilesize_x, tilesize_y, ext)
        out = os.path.join(out_dir, out_name)

    # Add log file handler
    log_file = args.log_file
    log_dir = args.log_dir
    if not log_file:
        if not log_dir:
            log_dir = os.path.dirname(out)
        log_name = os.path.basename(out).replace('.shp', '_log.txt')
        log_file = os.path.join(log_dir, log_name)

    logger = create_logger(__name__, 'fh',
                           handler_level='DEBUG',
                           filename=log_file)

    otb_lsms(img=image_source,
             out=out,
             mode=mode,
             spatialr=spatialr,
             ranger=ranger,
             minsize=minsize,
             tilesize_x=tilesize_x,
             tilesize_y=tilesize_y,
             overwrite=overwrite)
