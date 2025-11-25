#!/bin/bash
# Helper script to mount/unmount reef_harddisk
# Usage: ./mount_reef_harddisk.sh [mount|unmount|status]

set -e

MOUNT_POINT="$HOME/reef_harddisk"
REMOTE_PATH="tao@192.168.2.1:/media/reef/harddisk"
SSH_KEY="$HOME/.ssh/id_rsa"

# Get current user's UID and GID
USER_UID=$(id -u)
USER_GID=$(id -g)

function mount_disk() {
    echo "Mounting reef_harddisk..."
    
    # Check if already mounted
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Already mounted at $MOUNT_POINT"
        return 0
    fi
    
    # Ensure mount point exists
    mkdir -p "$MOUNT_POINT"
    
    # Test SSH connection first
    echo "Testing SSH connection..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$(echo $REMOTE_PATH | cut -d: -f1)" "echo 'Connection test'" >/dev/null 2>&1; then
        echo "⚠ Warning: SSH connection test failed. You may need to set up SSH key authentication."
        echo "Run: ssh-copy-id $(echo $REMOTE_PATH | cut -d: -f1)"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Mount using sshfs
    sshfs "$REMOTE_PATH" "$MOUNT_POINT" \
        -o default_permissions,IdentityFile="$SSH_KEY",uid=$USER_UID,gid=$USER_GID
    
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Successfully mounted to $MOUNT_POINT"
        echo "  Remote: $REMOTE_PATH"
        echo "  Local:  $MOUNT_POINT"
        df -h | grep reef_harddisk
    else
        echo "✗ Mount failed!"
        exit 1
    fi
}

function unmount_disk() {
    echo "Unmounting reef_harddisk..."
    
    if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Not mounted"
        return 0
    fi
    
    fusermount -u "$MOUNT_POINT"
    
    if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        echo "✓ Successfully unmounted"
    else
        echo "✗ Unmount failed! Trying force unmount..."
        fusermount -uz "$MOUNT_POINT"
        if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
            echo "✓ Force unmount successful"
        else
            echo "✗ Force unmount also failed. You may need to close files/programs using the mount."
            exit 1
        fi
    fi
}

function show_status() {
    echo "Reef Harddisk Mount Status"
    echo "=========================="
    
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        echo "Status: ✓ MOUNTED"
        echo ""
        echo "Mount point: $MOUNT_POINT"
        echo "Remote path: $REMOTE_PATH"
        echo ""
        echo "Disk usage:"
        df -h | grep reef_harddisk
        echo ""
        echo "Mount details:"
        mount | grep reef_harddisk
    else
        echo "Status: ✗ NOT MOUNTED"
        echo ""
        echo "Mount point: $MOUNT_POINT"
        echo "Remote path: $REMOTE_PATH"
        echo ""
        echo "To mount, run: $0 mount"
    fi
}

# Main script logic
case "${1:-status}" in
    mount)
        mount_disk
        ;;
    unmount|umount)
        unmount_disk
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [mount|unmount|status]"
        echo ""
        echo "Commands:"
        echo "  mount    - Mount the reef_harddisk"
        echo "  unmount  - Unmount the reef_harddisk"
        echo "  status   - Show mount status (default)"
        exit 1
        ;;
esac

