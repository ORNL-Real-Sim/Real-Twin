'''
##############################################################
# Created Date: Monday, December 23rd 2024
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''

from test_setup import add_pkg_to_sys_path
add_pkg_to_sys_path("realtwin")

from realtwin.utils_lib.check_env import is_sumo_installed, is_vissim_installed, is_aimsun_installed


def test_is_sumo_installed():
    assert is_sumo_installed()
