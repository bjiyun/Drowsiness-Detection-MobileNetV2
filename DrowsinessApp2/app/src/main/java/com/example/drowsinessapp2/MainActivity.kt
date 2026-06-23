package com.example.drowsinessapp2

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Size
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var detailText: TextView
    private lateinit var resetButton: Button

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val isProcessing = AtomicBoolean(false)

    private var detector: DrowsinessDetector? = null
    private lateinit var toneGenerator: ToneGenerator
    private var lastAlarmTime = 0L

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(this, "카메라 권한이 필요합니다.", Toast.LENGTH_LONG).show()
                statusText.text = "카메라 권한 없음"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        statusText = findViewById(R.id.statusText)
        detailText = findViewById(R.id.detailText)
        resetButton = findViewById(R.id.resetButton)

        toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)

        try {
            detector = DrowsinessDetector(this)
        } catch (e: Exception) {
            statusText.text = "모델 로드 실패"
            detailText.text = e.message ?: "assets 폴더의 모델 파일을 확인하세요."
            return
        }

        resetButton.setOnClickListener {
            detector?.resetCalibration()
            statusText.text = "정상 얼굴 등록 중"
            detailText.text = "3초 동안 정면을 바라보고 눈을 자연스럽게 뜬 상태를 유지해주세요."
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(480, 360))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val localDetector = detector

                if (localDetector == null) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                if (!isProcessing.compareAndSet(false, true)) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                try {
                    val bitmap = imageProxy.toUprightBitmap()
                    val result = localDetector.analyze(bitmap)

                    runOnUiThread {
                        showResult(result)
                    }

                } catch (e: Exception) {
                    runOnUiThread {
                        statusText.text = "분석 오류"
                        detailText.text = e.message ?: "알 수 없는 오류"
                    }
                } finally {
                    imageProxy.close()
                    isProcessing.set(false)
                }
            }

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalysis
                )

            } catch (e: Exception) {
                statusText.text = "카메라 실행 실패"
                detailText.text = e.message ?: "전면 카메라를 확인하세요."
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun showResult(result: DetectionResult) {
        when (result.status) {
            DetectionStatus.CALIBRATING -> {
                statusText.text = "초기화 중"
            }

            DetectionStatus.NORMAL -> {
                statusText.text = "정상 상태"
            }

            DetectionStatus.DROWSY -> {
                statusText.text = "졸음 감지!"
                playAlarm()
            }

            DetectionStatus.NO_FACE -> {
                statusText.text = "얼굴 없음"
            }
        }

        detailText.text = """
            ${result.message}
            EAR: ${"%.3f".format(result.ear)} / 기준: ${"%.3f".format(result.earThreshold)}
            MAR: ${"%.3f".format(result.mar)} / 기준: ${"%.3f".format(result.marThreshold)}
            눈 감김 확률: ${"%.2f".format(result.eyeClosedProb)}
            하품 확률: ${"%.2f".format(result.yawnProb)}
            종합 점수: ${"%.2f".format(result.fusionScore)}
            졸음 감지 횟수: ${result.drowsyCount}
        """.trimIndent()
    }

    private fun playAlarm() {
        val now = System.currentTimeMillis()

        if (now - lastAlarmTime < 1500L) return

        lastAlarmTime = now
        toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, 800)
        vibrate()
    }

    private fun vibrate() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val manager = getSystemService(VibratorManager::class.java)
                manager.defaultVibrator.vibrate(
                    VibrationEffect.createOneShot(
                        300,
                        VibrationEffect.DEFAULT_AMPLITUDE
                    )
                )
            } else {
                @Suppress("DEPRECATION")
                val vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator.vibrate(
                        VibrationEffect.createOneShot(
                            300,
                            VibrationEffect.DEFAULT_AMPLITUDE
                        )
                    )
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(300)
                }
            }
        } catch (_: Exception) {
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        toneGenerator.release()
        cameraExecutor.shutdown()
    }
}