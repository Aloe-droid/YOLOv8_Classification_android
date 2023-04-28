package com.example.yolov8n_cls

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import java.util.*
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var textView: TextView
    private lateinit var dataProcess: DataProcess
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var session: OrtSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        textView = findViewById(R.id.textView)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setPermissions()
        dataProcess = DataProcess()
        load()
        setCamera()
    }

    private fun setCamera() {
        //카메라 제공 객체
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        //전체 화면
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        // 후면 카메라
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY).build()

        // 16:9 화면으로 받아옴
        val preview = Preview.Builder().setResolutionSelector(resolutionSelector).build()

        // preview 에서 받아와서 previewView 에 보여준다.
        preview.setSurfaceProvider(previewView.surfaceProvider)

        //분석 중이면 그 다음 화면이 대기중인 것이 아니라 계속 받아오는 화면으로 새로고침 함. 분석이 끝나면 그 최신 사진을 다시 분석
        val analysis = ImageAnalysis.Builder().setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()

        //여기서 it == imageProxy 객체이다.
        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }

        // 카메라의 수명 주기를 메인 액티비티에 귀속
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    // 이미지 처리 s21 Ultra == 35ms ~ 42ms
    private fun imageProcess(imageProxy: ImageProxy) {
        val bitmap = dataProcess.imageToBitmap(imageProxy)
        val floatBuffer = dataProcess.bitmapToFloatBuffer(bitmap)
        val inputName = session.inputNames.iterator().next() // session 이름
        //모델의 요구 입력값 [1 3 224 224] [배치 사이즈, 픽셀(RGB), 너비, 높이], 모델마다 크기는 다를 수 있음.
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))
        val outputs = resultTensor.get(0).value as Array<*> // [1 1000]
        val index = dataProcess.getHighConf(outputs)
        val name = dataProcess.getClassName(index)

        runOnUiThread {
            name?.let { textView.text = it }
        }
    }

    private fun load() {
        dataProcess.loadModel(this) // onnx 모델 불러오기
        dataProcess.loadLabel(this) // coco txt 파일 불러오기

        ortEnvironment = OrtEnvironment.getEnvironment()
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == 1) {
            grantResults.forEach {
                if (it != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "권한을 허용하지 않으면 사용할 수 없습니다!", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setPermissions() {
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.CAMERA)

        permissions.forEach {
            if (ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions.toTypedArray(), 1)
            }
        }
    }
}