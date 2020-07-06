import os
from rindcalc import naip


def index(input_naip, out_naip, index='ARVI'):
    """
    Wrapper function to create the desired vegetation index for training using
    rindcalc.
    Args:
        input_naip: Input NAIP tile
        out_naip: Output ARVI index raster
        index: Which vegatation index to create
    """
    if not os.path.exists(input_naip):
        raise IOError('Path not found.')
    if os.path.exists(input_naip):
        i = getattr(naip, index)(input_naip, out_naip)

    print('Finished')
