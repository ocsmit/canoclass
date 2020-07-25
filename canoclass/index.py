import os
from rindcalc import naip


def index(input_naip, out_naip, index='ARVI'):
    """
    Wrapper function to create the desired vegetation index for training using
    rindcalc. Use for NAIP only, for other imagery directly use Rindcalc.

    Parameters
    ----------
        input_naip: str, filepath
            Input NAIP tile
        out_naip: str, filepath
            Output ARVI index raster
        index : str
            Which vegatation index to create
    """
    if not os.path.exists(input_naip):
        raise IOError('Path not found.')
    if os.path.exists(input_naip):
        i = getattr(naip, index)(input_naip, out_naip)

    print('Finished')
