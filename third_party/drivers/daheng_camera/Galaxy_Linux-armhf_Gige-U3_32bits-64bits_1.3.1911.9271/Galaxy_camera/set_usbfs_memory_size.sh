#!/bin/sh
# ***************************************************************************************
#
#
# ***************************************************************************************

DISTRIBUTION="unknown"

if [ -f /etc/lsb-release ]; then
    if [ -f /lib/lsb/init-functions ]; then
        DISTRIBUTION=`grep DISTRIB_ID /etc/lsb-release | sed -e 's/DISTRIB_ID=//'`
    fi
elif [ -f /etc/os-release ]; then
    if [ -f /lib/lsb/init-functions ]; then
        DISTRIBUTION=`grep -w NAME /etc/os-release | sed 's/[^"]*"\([^"]*\)"/\1/'`
        if [ "$DISTRIBUTION" == "Debian GNU/Linux" ] ; then
            # Debian is close enough for the purposes of this script
            DISTRIBUTION="Ubuntu"
        fi
    fi
fi

# do as sudo or root
if [ $(id -u) -ne 0 ] ; then
    echo "User has insufficient privileges. Please try sudo command."
    exit 1
fi

# increase usb memory
echo "Setting usbfs memory size to 1000"
sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'

if grep --silent usbcore.usbfs_memory_mb=1000 /etc/default/grub; then
    echo "" > /dev/null
else
    if grep --silent ^GRUB_CMDLINE_LINUX..usbcore.usbfs_memory_mb /etc/default/grub; then
    	sed -i 's/^GRUB_CMDLINE_LINUX..usbcore.usbfs_memory_mb.*/GRUB_CMDLINE_LINUX="usbcore.usbfs_memory_mb=1000"/g' /etc/default/grub
    else
    	echo "" >> /etc/default/grub
    	echo "GRUB_CMDLINE_LINUX=\"usbcore.usbfs_memory_mb=1000\"" >> /etc/default/grub
    fi

    if [ "$DISTRIBUTION" = "Ubuntu" ]; then
        update-grub
    else
        grub2-mkconfig -o /boot/grub2/grub.cfg
    fi
fi




