#!/bin/zsh
# Task-owned, bounded VM/client; never select or start Docker Desktop services.
set -e
recipe_root="${0:A:h}"
export PATH="$recipe_root/.cache/vm-runtime/prefix/bin:$PATH"
recipe_vm_root="${QWEN_VM_ROOT:-$HOME/.local/share/q38eval}"
export COLIMA_HOME="$recipe_vm_root/c"
export LIMA_HOME="$recipe_vm_root/l"
export DOCKER_CONFIG="$recipe_vm_root/d"
if [[ "$1" == docker ]]; then
  shift
  exec docker --context colima-qwen38 "$@"
elif [[ "$1" == start ]]; then
  recipe_vm_memory="${QWEN_VM_MEMORY_GB:-2}"
  case "$recipe_vm_memory" in 2|3) ;; *) print -u2 'Task VM memory must be 2 or 3 GB'; exit 2 ;; esac
  mkdir -p "$recipe_root/runs/code-jobs"
  colima start qwen38 --vm-type vz --arch aarch64 --cpus 2 --memory "$recipe_vm_memory" --disk 8 --root-disk 8 --runtime docker --activate=false --ssh-config=false --ssh-agent=false --mount "$recipe_root/runs/code-jobs" --mount-type virtiofs --port-forwarder none --template=false --binfmt=false --dns 1.1.1.1 --dns 8.8.8.8
  # The pinned Ubuntu image lacked /etc/resolv.conf; configure this guest only.
  printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\noptions timeout:2 attempts:2\n' | colima -p qwen38 ssh -- sudo sh -c 'cat > /etc/resolv.conf.qwen; mv /etc/resolv.conf.qwen /etc/resolv.conf'
else
  exec colima -p qwen38 "$@"
fi
