#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "two command line arguments expected" >&2
fi

PREFIX="${1}"
RES_DIR="${2}"

if [[ -z ${PREFIX} ]]; then
    echo "no prefix specified" >&2
fi

typeset -i CNT=0

if [[ -d "${RES_DIR}" ]]; then

    USE_SVGO=false ; command -v svgo > /dev/null 2>&1 && USE_SVGO=true

    USE_SVGCLEANER=false ; command -v svgcleaner > /dev/null 2>&1 && USE_SVGCLEANER=true

    USE_YUICOMP=false ; command -v yuicompressor > /dev/null 2>&1 && USE_YUICOMP=false

    MANIFEST="${RES_DIR}/$(basename -- "${RES_DIR}").gresource.xml"

    MANIFEST_BASE="$(basename -- ${MANIFEST})"

    rm -f -- "${MANIFEST}"

    printf '<?xml version="1.0" encoding="UTF-8"?>\n' >> "${MANIFEST}"
    printf '<gresources>\n' >> "${MANIFEST}"
    printf '\t<gresource prefix="%s">\n' "${PREFIX}" >> "${MANIFEST}"

    while read RES_FILE; do

        ATTRS='compressed="true"'
        RES_FILE="${RES_FILE#./}"

        RES_FILE_BASE="$(basename -- ${RES_FILE})"

        if [[ ${RES_FILE[1]} != '.' && "${MANIFEST_BASE}" != "${RES_FILE_BASE}" ]]; then

            RES_FILE_PATH="${RES_DIR}/${RES_FILE}"

            RES_FILE_EXT="$(echo ${RES_FILE##*.} | tr '[:upper:]' '[:lower:]')"

            if [[ "${RES_FILE_EXT}" != gresource ]]; then

                if [[ "${RES_FILE_EXT}" == svg ]]; then
                    if $USE_SVGO; then
                        printf "optimizing '%s'...\n" "${RES_FILE}"
                        svgo -- "${RES_FILE_PATH}"
                        echo
                    fi
                fi



                if $USE_YUICOMP && [[ "${RES_FILE_EXT}" == css ]]; then
                    printf "minifying '%s'...\n" "${RES_FILE}"
                    yuicompressor -o "${RES_FILE_PATH}.tmp" --type css \
                        -- "${RES_FILE_PATH}"
                    mv -f -- "${RES_FILE_PATH}.tmp" "${RES_FILE_PATH}"
                    echo
                fi

                unset ATTR_STRIP
                if [[ "${RES_FILE_EXT}" =~ ^(xml|svg|htm|html|ui)$ ]]; then
                    ATTRS+=' preprocess="xml-stripblanks"'
                fi

                printf '\t\t<file alias="%s" %s>%s</file>\n' \
                    "${RES_FILE}" \
                    "${ATTRS}" \
                    "${RES_FILE}" >> ${MANIFEST}

                printf "%s' processed (%d)\n" "${RES_FILE}" $((++CNT))
            fi
        fi

    done < <(cd -- "${RES_DIR}" && find . -type f)

    if [[ $CNT -eq 0 ]]; then
        printf "no resource files found in '%s'" "${RES_DIR}"
        rm -f -- "${MANIFEST}"
    fi

    printf '\t</gresource>\n' >> "${MANIFEST}"
    printf '</gresources>\n' >> "${MANIFEST}"


   (cd -- ${RES_DIR} && glib-compile-resources -- $(basename -- "${MANIFEST}"))

    if [[ -f "${MANIFEST%.xml}" ]]; then
       printf "created '%s':\n" "${MANIFEST%.xml}"
       gresource details "${MANIFEST%.xml}"
    fi
else
    printf "'%s': no such directory" "${RES_DIR}"
fi
