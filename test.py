tensor = tf.image.decode_image(contents=image, channels=3)
                tensor = tf.cast(x=tensor, dtype=tf.float32)
                decoded = tensor.eval()
                decoded = np.reshape(a=decoded, newshape=(1, decoded.shape[0], decoded.shape[1], decoded.shape[2]))
                tensor = tf.image.resize_bicubic(images=decoded, size=(128, 128))
                # tensor = tf.image.resize_image_with_crop_or_pad(image=decoded, target_width=128, target_height=128)
                tensor = tf.reshape(tensor=tensor, shape=(tensor.shape[1], tensor.shape[2], tensor.shape[3]))
                decoded = tensor.eval()