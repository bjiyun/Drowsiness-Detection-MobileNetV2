package com.example.drowsinessapp2

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

class TFLiteBinaryClassifier(
    context: Context,
    private val assetName: String
) : AutoCloseable {

    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val inputDataType: DataType
    private val outputDataType: DataType

    private val inputHeight: Int
    private val inputWidth: Int
    private val inputChannels: Int

    init {
        val modelBuffer = loadModelFile(context, assetName)

        interpreter = Interpreter(
            modelBuffer,
            Interpreter.Options().apply {
                setNumThreads(2)
            }
        )

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        inputShape = inputTensor.shape()
        inputDataType = inputTensor.dataType()
        outputDataType = outputTensor.dataType()

        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        inputChannels = inputShape[3]
    }

    fun predict(bitmap: Bitmap): Float {
        val resized = Bitmap.createScaledBitmap(
            bitmap,
            inputWidth,
            inputHeight,
            true
        )

        val inputBuffer = bitmapToInputBuffer(resized)

        val outputTensor = interpreter.getOutputTensor(0)
        val outputSize = outputTensor.numElements()

        val outputBuffer = ByteBuffer
            .allocateDirect(outputSize * bytesPerElement(outputDataType))
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()

        val result = when (outputDataType) {
            DataType.FLOAT32 -> {
                outputBuffer.float
            }

            DataType.UINT8 -> {
                val raw = outputBuffer.get().toInt() and 0xFF
                val params = outputTensor.quantizationParams()

                if (params.scale > 0f) {
                    params.scale * (raw - params.zeroPoint)
                } else {
                    raw / 255f
                }
            }

            DataType.INT8 -> {
                val raw = outputBuffer.get().toInt()
                val params = outputTensor.quantizationParams()

                if (params.scale > 0f) {
                    params.scale * (raw - params.zeroPoint)
                } else {
                    (raw + 128) / 255f
                }
            }

            else -> {
                throw IllegalStateException("지원하지 않는 출력 타입: $outputDataType")
            }
        }

        return result.coerceIn(0f, 1f)
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer
            .allocateDirect(inputWidth * inputHeight * inputChannels * bytesPerElement(inputDataType))
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputWidth * inputHeight)

        bitmap.getPixels(
            pixels,
            0,
            inputWidth,
            0,
            0,
            inputWidth,
            inputHeight
        )

        val inputTensor = interpreter.getInputTensor(0)
        val params = inputTensor.quantizationParams()

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            putValue(buffer, r, inputDataType, params)
            putValue(buffer, g, inputDataType, params)
            putValue(buffer, b, inputDataType, params)
        }

        buffer.rewind()
        return buffer
    }

    private fun putValue(
        buffer: ByteBuffer,
        value: Float,
        type: DataType,
        params: Tensor.QuantizationParams
    ) {
        when (type) {
            DataType.FLOAT32 -> {
                buffer.putFloat(value)
            }

            DataType.UINT8 -> {
                val q = if (params.scale > 0f) {
                    ((value / params.scale) + params.zeroPoint)
                        .roundToInt()
                        .coerceIn(0, 255)
                } else {
                    (value * 255f)
                        .roundToInt()
                        .coerceIn(0, 255)
                }

                buffer.put(q.toByte())
            }

            DataType.INT8 -> {
                val q = if (params.scale > 0f) {
                    ((value / params.scale) + params.zeroPoint)
                        .roundToInt()
                        .coerceIn(-128, 127)
                } else {
                    ((value * 255f) - 128f)
                        .roundToInt()
                        .coerceIn(-128, 127)
                }

                buffer.put(q.toByte())
            }

            else -> {
                throw IllegalStateException("지원하지 않는 입력 타입: $type")
            }
        }
    }

    private fun bytesPerElement(type: DataType): Int {
        return when (type) {
            DataType.FLOAT32 -> 4
            DataType.INT32 -> 4
            DataType.UINT8 -> 1
            DataType.INT8 -> 1
            else -> 4
        }
    }

    private fun loadModelFile(
        context: Context,
        assetName: String
    ): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(assetName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    override fun close() {
        interpreter.close()
    }
}