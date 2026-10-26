#!/bin/sh
# Compiles a theme directory into a GResource bundle:
#
#     compile-gresources.sh <PREFIX> <RESOURCE DIRECTORY> [OUTPUT FILE]
#
# The output defaults to <RESOURCE DIRECTORY>/<NAME>.gresource. Source files
# are never changed: they are copied to a temporary directory, where SVG
# icons are optimized if svgo is installed, and listed in sorted order, so
# that the bundle only changes when the sources do.

set -eu

usage() {
    echo "usage: ${0##*/} <PREFIX> <RESOURCE DIRECTORY> [OUTPUT FILE]" >&2
    exit 1
}

[ $# -ge 2 ] && [ $# -le 3 ] || usage

prefix=$1
res_dir=${2%/}
output=${3:-$res_dir/${res_dir##*/}.gresource}

if [ -z "$prefix" ]; then
    echo "no prefix specified" >&2
    exit 1
fi

if [ ! -d "$res_dir" ]; then
    printf "'%s': no such directory\n" "$res_dir" >&2
    exit 1
fi

case $output in
    /*) ;;
    *) output=$PWD/$output ;;
esac

staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
trap 'exit 130' INT TERM

xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

mkdir -- "$staging/files"
manifest=$staging/manifest.xml

{
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<gresources>\n'
    printf '\t<gresource prefix="%s">\n' "$(xml_escape "$prefix")"

    # skip hidden files, editor backups and bundles
    (cd -- "$res_dir" && find . -type f ! -path '*/.*' ! -name '*~' \
        ! -name '*.gresource' ! -name '*.gresource.xml' | LC_ALL=C sort) |
        while IFS= read -r file; do
            file=${file#./}
            mkdir -p -- "$staging/files/$(dirname -- "$file")"
            cp -- "$res_dir/$file" "$staging/files/$file"
            printf '\t\t<file compressed="true">%s</file>\n' "$(xml_escape "$file")"
        done

    printf '\t</gresource>\n'
    printf '</gresources>\n'
} > "$manifest"

if ! grep -q '<file ' "$manifest"; then
    printf "no resource files found in '%s'\n" "$res_dir" >&2
    exit 1
fi

if command -v svgo > /dev/null 2>&1; then
    svgo --quiet --recursive --folder "$staging/files"
fi

glib-compile-resources --sourcedir="$staging/files" --target="$output" "$manifest"

printf "created '%s':\n" "$output"
gresource list "$output"
