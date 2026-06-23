package com.example.drowsinessapp2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.os.SystemClock
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max

enum class DetectionStatus {
    CALIBRATING,
    NORMAL,
    DROWSY,
    NO_FACE
}

data class DetectionResult(
    val status: DetectionStatus,
    val message: String,
    val ear: Float,
    val mar: Float,
    val earThreshold: Float,
    val marThreshold: Float,
    val eyeClosedProb: Float,
    val yawnProb: Float,
    val fusionScore: Float,
    val drowsyCount: Int
)

class DrowsinessDetector(
    context: Context
) : AutoCloseable {

    private val faceLandmarker: FaceLandmarker

    private val eyeModel = TFLiteBinaryClassifier(
        context,
        "eye_model.tflite"
    )

    private val mouthModel = TFLiteBinaryClassifier(
        context,
        "mouth_model.tflite"
    )

    private var calibrationMode = true
    private var calibrationStartMs = SystemClock.elapsedRealtime()

    private val normalEarValues = mutableListOf<Float>()
    private val normalMarValues = mutableListOf<Float>()

    private var personalEarThreshold = DEFAULT_EAR_THRESHOLD
    private var personalMarThreshold = DEFAULT_MAR_THRESHOLD

    private var eyeClosedStartMs: Long? = null
    private var yawnStartMs: Long? = null
    private var fusionStartMs: Long? = null

    private var drowsyState = false
    private var drowsyCount = 0

    private var eyeClosedProb = 0f
    private var yawnProb = 0f
    private var fusionScore = 0f

    private var frameCount = 0

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()

        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(
            context,
            options
        )
    }

    @Synchronized
    fun analyze(bitmap: Bitmap): DetectionResult {
        frameCount++

        val now = SystemClock.elapsedRealtime()

        val mpImage = BitmapImageBuilder(bitmap).build()
        val faceResult = faceLandmarker.detect(mpImage)

        if (faceResult.faceLandmarks().isEmpty()) {
            resetTimers()

            return DetectionResult(
                status = DetectionStatus.NO_FACE,
                message = "얼굴이 화면에 없습니다.",
                ear = 0f,
                mar = 0f,
                earThreshold = personalEarThreshold,
                marThreshold = personalMarThreshold,
                eyeClosedProb = eyeClosedProb,
                yawnProb = yawnProb,
                fusionScore = fusionScore,
                drowsyCount = drowsyCount
            )
        }

        val landmarks = faceResult.faceLandmarks()[0]

        val width = bitmap.width
        val height = bitmap.height

        val leftEyeEarPoints = getPoints(
            landmarks,
            LEFT_EYE_EAR,
            width,
            height
        )

        val rightEyeEarPoints = getPoints(
            landmarks,
            RIGHT_EYE_EAR,
            width,
            height
        )

        val mouthMarPoints = getPoints(
            landmarks,
            MOUTH_MAR,
            width,
            height
        )

        val leftEar = eyeAspectRatio(leftEyeEarPoints)
        val rightEar = eyeAspectRatio(rightEyeEarPoints)

        val ear = (leftEar + rightEar) / 2f
        val mar = mouthAspectRatio(mouthMarPoints)

        if (calibrationMode) {
            normalEarValues.add(ear)
            normalMarValues.add(mar)

            val elapsedSecond = (now - calibrationStartMs) / 1000f
            val remain = max(0f, INIT_TIME - elapsedSecond)

            if (elapsedSecond >= INIT_TIME) {
                finishCalibration()
            }

            return DetectionResult(
                status = if (calibrationMode) {
                    DetectionStatus.CALIBRATING
                } else {
                    DetectionStatus.NORMAL
                },
                message = if (calibrationMode) {
                    "정면을 보고 눈을 뜨고 입을 닫아주세요. 남은 시간: ${"%.1f".format(remain)}초"
                } else {
                    "초기화 완료. 졸음 감지를 시작합니다."
                },
                ear = ear,
                mar = mar,
                earThreshold = personalEarThreshold,
                marThreshold = personalMarThreshold,
                eyeClosedProb = eyeClosedProb,
                yawnProb = yawnProb,
                fusionScore = fusionScore,
                drowsyCount = drowsyCount
            )
        }

        val leftEyeCrop = cropByPoints(
            bitmap,
            getPoints(landmarks, LEFT_EYE_CROP, width, height),
            12
        )

        val rightEyeCrop = cropByPoints(
            bitmap,
            getPoints(landmarks, RIGHT_EYE_CROP, width, height),
            12
        )

        val mouthCrop = cropByPoints(
            bitmap,
            getPoints(landmarks, MOUTH_CROP, width, height),
            22
        )

        if (frameCount % DL_INTERVAL == 0) {
            val eyeProbList = mutableListOf<Float>()

            if (leftEyeCrop != null) {
                val raw = eyeModel.predict(leftEyeCrop)

                val closedProb = if (EYE_OUTPUT_IS_CLOSED_PROB) {
                    raw
                } else {
                    1f - raw
                }

                eyeProbList.add(closedProb)
            }

            if (rightEyeCrop != null) {
                val raw = eyeModel.predict(rightEyeCrop)

                val closedProb = if (EYE_OUTPUT_IS_CLOSED_PROB) {
                    raw
                } else {
                    1f - raw
                }

                eyeProbList.add(closedProb)
            }

            if (eyeProbList.isNotEmpty()) {
                eyeClosedProb = eyeProbList.average()
                    .toFloat()
                    .coerceIn(0f, 1f)
            }

            if (mouthCrop != null) {
                val raw = mouthModel.predict(mouthCrop)

                yawnProb = if (MOUTH_OUTPUT_IS_YAWN_PROB) {
                    raw
                } else {
                    1f - raw
                }

                yawnProb = yawnProb.coerceIn(0f, 1f)
            }
        }

        val earRuleScore = if (ear < personalEarThreshold) 1f else 0f
        val marRuleScore = if (mar > personalMarThreshold) 1f else 0f

        fusionScore =
            EYE_MODEL_WEIGHT * eyeClosedProb +
                    MOUTH_MODEL_WEIGHT * yawnProb +
                    EAR_RULE_WEIGHT * earRuleScore +
                    MAR_RULE_WEIGHT * marRuleScore

        if (ear < personalEarThreshold || eyeClosedProb >= EYE_CLOSED_PROB_THRESHOLD) {
            if (eyeClosedStartMs == null) {
                eyeClosedStartMs = now
            }
        } else {
            eyeClosedStartMs = null
        }

        if (mar > personalMarThreshold || yawnProb >= YAWN_PROB_THRESHOLD) {
            if (yawnStartMs == null) {
                yawnStartMs = now
            }
        } else {
            yawnStartMs = null
        }

        val fusionCondition =
            fusionScore >= FINAL_THRESHOLD &&
                    (earRuleScore == 1f || marRuleScore == 1f)

        if (fusionCondition) {
            if (fusionStartMs == null) {
                fusionStartMs = now
            }
        } else {
            fusionStartMs = null
        }

        var isDrowsy = false
        var message = "정상 상태입니다."

        if (eyeClosedStartMs != null &&
            now - eyeClosedStartMs!! >= EYE_CLOSED_TIME_MS
        ) {
            isDrowsy = true
            message = "눈을 오래 감고 있습니다."
        }

        if (yawnStartMs != null &&
            now - yawnStartMs!! >= YAWN_TIME_MS
        ) {
            isDrowsy = true
            message = "하품이 감지되었습니다."
        }

        if (fusionStartMs != null &&
            now - fusionStartMs!! >= FUSION_HOLD_TIME_MS
        ) {
            isDrowsy = true
            message = "졸음 상태로 판단되었습니다."
        }

        if (isDrowsy) {
            if (!drowsyState) {
                drowsyCount++
                drowsyState = true
            }
        } else {
            drowsyState = false
        }

        return DetectionResult(
            status = if (isDrowsy) {
                DetectionStatus.DROWSY
            } else {
                DetectionStatus.NORMAL
            },
            message = message,
            ear = ear,
            mar = mar,
            earThreshold = personalEarThreshold,
            marThreshold = personalMarThreshold,
            eyeClosedProb = eyeClosedProb,
            yawnProb = yawnProb,
            fusionScore = fusionScore,
            drowsyCount = drowsyCount
        )
    }

    fun resetCalibration() {
        calibrationMode = true
        calibrationStartMs = SystemClock.elapsedRealtime()

        normalEarValues.clear()
        normalMarValues.clear()

        personalEarThreshold = DEFAULT_EAR_THRESHOLD
        personalMarThreshold = DEFAULT_MAR_THRESHOLD

        eyeClosedProb = 0f
        yawnProb = 0f
        fusionScore = 0f

        resetTimers()
    }

    private fun finishCalibration() {
        if (normalEarValues.size < 5 || normalMarValues.size < 5) {
            personalEarThreshold = DEFAULT_EAR_THRESHOLD
            personalMarThreshold = DEFAULT_MAR_THRESHOLD
        } else {
            val baseEar = normalEarValues.average().toFloat()
            val baseMar = normalMarValues.average().toFloat()

            personalEarThreshold = (baseEar * 0.65f)
                .coerceIn(0.12f, 0.30f)

            personalMarThreshold = (baseMar * 1.8f)
                .coerceIn(0.20f, 1.20f)
        }

        calibrationMode = false
    }

    private fun resetTimers() {
        eyeClosedStartMs = null
        yawnStartMs = null
        fusionStartMs = null
        drowsyState = false
    }

    private fun getPoints(
        landmarks: List<NormalizedLandmark>,
        indices: IntArray,
        width: Int,
        height: Int
    ): List<PointF> {
        return indices.map { index ->
            val landmark = landmarks[index]

            PointF(
                landmark.x() * width,
                landmark.y() * height
            )
        }
    }

    private fun cropByPoints(
        bitmap: Bitmap,
        points: List<PointF>,
        padding: Int
    ): Bitmap? {
        if (points.isEmpty()) return null

        val minX = points.minOf { it.x }
        val maxX = points.maxOf { it.x }
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }

        val x1 = floor(minX - padding)
            .toInt()
            .coerceAtLeast(0)

        val y1 = floor(minY - padding)
            .toInt()
            .coerceAtLeast(0)

        val x2 = floor(maxX + padding)
            .toInt()
            .coerceAtMost(bitmap.width)

        val y2 = floor(maxY + padding)
            .toInt()
            .coerceAtMost(bitmap.height)

        val cropWidth = x2 - x1
        val cropHeight = y2 - y1

        if (cropWidth <= 2 || cropHeight <= 2) {
            return null
        }

        return Bitmap.createBitmap(
            bitmap,
            x1,
            y1,
            cropWidth,
            cropHeight
        )
    }

    private fun eyeAspectRatio(points: List<PointF>): Float {
        val a = distance(points[1], points[5])
        val b = distance(points[2], points[4])
        val c = distance(points[0], points[3])

        if (c == 0f) return 0f

        return (a + b) / (2f * c)
    }

    private fun mouthAspectRatio(points: List<PointF>): Float {
        val v1 = distance(points[1], points[7])
        val v2 = distance(points[2], points[6])
        val v3 = distance(points[3], points[5])
        val h = distance(points[0], points[4])

        if (h == 0f) return 0f

        return (v1 + v2 + v3) / (2f * h)
    }

    private fun distance(
        p1: PointF,
        p2: PointF
    ): Float {
        return hypot(
            p1.x - p2.x,
            p1.y - p2.y
        )
    }

    override fun close() {
        faceLandmarker.close()
        eyeModel.close()
        mouthModel.close()
    }

    companion object {
        private const val INIT_TIME = 3.0f

        private const val DEFAULT_EAR_THRESHOLD = 0.23f
        private const val DEFAULT_MAR_THRESHOLD = 0.60f

        private const val EYE_CLOSED_TIME_MS = 700L
        private const val YAWN_TIME_MS = 1000L
        private const val FUSION_HOLD_TIME_MS = 700L

        private const val EYE_MODEL_WEIGHT = 0.20f
        private const val MOUTH_MODEL_WEIGHT = 0.15f
        private const val EAR_RULE_WEIGHT = 0.45f
        private const val MAR_RULE_WEIGHT = 0.20f

        private const val FINAL_THRESHOLD = 0.50f

        private const val EYE_CLOSED_PROB_THRESHOLD = 0.60f
        private const val YAWN_PROB_THRESHOLD = 0.60f

        private const val EYE_OUTPUT_IS_CLOSED_PROB = true
        private const val MOUTH_OUTPUT_IS_YAWN_PROB = false

        private const val DL_INTERVAL = 5

        private val LEFT_EYE_EAR = intArrayOf(
            33, 160, 158, 133, 153, 144
        )

        private val RIGHT_EYE_EAR = intArrayOf(
            362, 385, 387, 263, 373, 380
        )

        private val MOUTH_MAR = intArrayOf(
            61, 81, 13, 311, 308, 402, 14, 178
        )

        private val LEFT_EYE_CROP = intArrayOf(
            33, 133, 160, 158, 153, 144, 159, 145
        )

        private val RIGHT_EYE_CROP = intArrayOf(
            362, 263, 385, 387, 373, 380, 386, 374
        )

        private val MOUTH_CROP = intArrayOf(
            61, 291, 13, 14, 78, 308, 81, 178, 402, 311
        )
    }
}