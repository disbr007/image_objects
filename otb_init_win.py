import subprocess
from subprocess import PIPE

from misc_utils.logging_utils import create_logger

logger = create_logger(__name__, 'sh', 'DEBUG')

otb_max_ram_hint = 4096
otb_env_loc = r"C:\OTB-7.1.0-Win64\OTB-7.1.0-Win64\otbenv.bat"


def run_subprocess(command):
    logger.debug('Running subprocess: {}'.format(command))
    proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):  # replace '' with b'' for Python 3
        logger.info(line.decode())
    proc_err = ""
    for line in iter(proc.stderr.readline, b''):
        proc_err += line.decode()
    if proc_err:
        logger.info(proc_err)
    output, error = proc.communicate()
    logger.debug('Output: {}'.format(output.decode()))
    logger.debug('Err: {}'.format(error.decode()))


def otb_init_win(otb_env_loc=otb_env_loc,
                 otb_max_ram_hint=otb_max_ram_hint):
    otb_env_loc = otb_env_loc
    run_subprocess(otb_env_loc)
    logger.info('Setting OTB_MAX_RAM_HINT={}'.format(otb_max_ram_hint))
    run_subprocess('set OTB_MAX_RAM_HINT={}'.format(otb_max_ram_hint))
    logger.info('OTB Max Ram:')
    run_subprocess('echo %OTB_MAX_RAM_HINT%')


if __name__ == '__main__':
    otb_init_win()