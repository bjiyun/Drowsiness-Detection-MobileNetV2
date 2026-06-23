package com.example.drowsinessapp2

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

fun ImageProxy.toUprightBitmap(): Bitmap {
    val nv21 = yuv420ToNv21(this)

    val yuvImage = YuvImage(
        nv21,
        ImageFormat.NV21,
        width,
        height,
        null
    )

    val outputStream = ByteArrayOutputStream()

    yuvImage.compressToJpeg(
        Rect(0, 0, width, height),
        85,
        outputStream
    )

    val jpegBytes = outputStream.toByteArray()

    val bitmap = BitmapFactory.decodeByteArray(
        jpegBytes,
        0,
        jpegBytes.size
    )

    val matrix = Matrix()
    matrix.postRotate(imageInfo.rotationDegrees.toFloat())

    return Bitmap.createBitmap(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        matrix,
        true
    )
}

private fun yuv420ToNv21(image: ImageProxy): ByteArray {
    val width = image.width
    val height = image.height

    val yPlane = image.planes[0]
    val uPlane = image.planes[1]
    val vPlane = image.planes[2]

    val nv21 = ByteArray(width * height + width * height / 2)

    var index = 0

    val yBuffer = yPlane.buffer
    val yRowStride = yPlane.rowStride
    val yPixelStride = yPlane.pixelStride

    for (row in 0 until height) {
        for (col in 0 until width) {
            val yIndex = row * yRowStride + col * yPixelStride
            nv21[index++] = yBuffer.get(yIndex)
        }
    }

    val uBuffer = uPlane.buffer
    val vBuffer = vPlane.buffer

    val chromaHeight = height / 2
    val chromaWidth = width / 2

    val uRowStride = uPlane.rowStride
    val vRowStride = vPlane.rowStride
    val uPixelStride = uPlane.pixelStride
    val vPixelStride = vPlane.pixelStride

    for (row in 0 until chromaHeight) {
        for (col in 0 until chromaWidth) {
            val vIndex = row * vRowStride + col * vPixelStride
            val uIndex = row * uRowStride + col * uPixelStride

            nv21[index++] = vBuffer.get(vIndex)
            nv21[index++] = uBuffer.get(uIndex)
        }
    }

    return nv21
}