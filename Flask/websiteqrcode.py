import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('掃描的人是笨蛋(*ﾟ∀ﾟ*)')
qr.make()
img = qr.make_image(fill_color="white", back_color="violet")
img.save("theqrcode.png")