pkgname=paddle_main
pkgver=VERSION
pkgrel=1
pkgdesc="paddle helper main script for uniform local train and cluster train"
depends=()
sources=("${JUMBO_REPO}/packages/${pkgname}/paddle")
md5sums=("MD5")

jumbo_install() {
  cd "${srcdir}"
  DESTDIR="${pkgdir}/${JUMBO_ROOT}/bin"
  mkdir -p "${DESTDIR}"
  cp -r "${srcdir}/paddle" "$DESTDIR"
  chmod +x "$DESTDIR/paddle"
}

# vim:set ft=sh ts=2 sw=2 et:
