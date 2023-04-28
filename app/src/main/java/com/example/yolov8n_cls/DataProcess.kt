package com.example.yolov8n_cls

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer

class DataProcess {

    private lateinit var classes: Array<String>

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_SIZE = 224
        const val PIXEL_SIZE = 3
        const val FILE_NAME = "yolov8n-cls.onnx"
        const val LABEL_NAME = "yolov8n-cls.txt"
    }

//    // 여러 개 반환 추천하지 않음
//    fun classNames(classList: List<Int>): String {
//        var names = ""
//        classList.forEachIndexed { index, it ->
//            if (index != 0) {
//                names += ", "
//            }
//            names += classes[it]
//        }
//        return names
//    }

    fun getClassName(i: Int?): String? {
        return if (i != null) {
            classes[i]
        } else null
    }

//    // 여러 개 반환 추천하지 않음
//    fun dataConfThresh(outputs: Array<*>): List<Int> {
//        val confThresholds = 0.2f
//        val output = outputs[0] as FloatArray
//        return output.withIndex().filter { it.value >= confThresholds }.map { it.index }
//    }

    //제일 높은 값 하나만 반환
    fun getHighConf(outputs: Array<*>): Int? {
        val confThresholds = 0.6f
        val output = outputs[0] as FloatArray
        return output.withIndex().filter { it.value >= confThresholds }
            .maxByOrNull { it.value }?.index
    }

    fun imageToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val matrix = Matrix().apply { postRotate(90f) }
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        return Bitmap.createBitmap(
            scaledBitmap,
            0,
            0,
            scaledBitmap.width,
            scaledBitmap.height,
            matrix,
            true
        )
    }


    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imageSTD = 255f
        val buffer = FloatBuffer.allocate(BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE)
        buffer.rewind()

        val area = INPUT_SIZE * INPUT_SIZE
        val bitmapData = IntArray(area)
        bitmap.getPixels(
            bitmapData,
            0,
            bitmap.width,
            0,
            0,
            bitmap.width,
            bitmap.height
        ) //배열에 RGB 담기

        //하나씩 받아서 버퍼에 할당
        for (i in 0 until INPUT_SIZE - 1) {
            for (j in 0 until INPUT_SIZE - 1) {
                val idx = INPUT_SIZE * i + j
                val pixelValue = bitmapData[idx]
                // 위에서 부터 차례대로 R 값 추출, G 값 추출, B값 추출 -> 255로 나누어서 0~1 사이로 정규화
                buffer.put(idx, ((pixelValue shr 16 and 0xff) / imageSTD))
                buffer.put(idx + area, ((pixelValue shr 8 and 0xff) / imageSTD))
                buffer.put(idx + area * 2, ((pixelValue and 0xff) / imageSTD))
                //원리 bitmap == ARGB 형태의 32bit, R값의 시작은 16bit (16 ~ 23bit 가 R영역), 따라서 16bit 를 쉬프트
                //그럼 A값이 사라진 RGB 값인 24bit 가 남는다. 이후 255와 AND 연산을 통해 맨 뒤 8bit 인 R값만 가져오고, 255로 나누어 정규화를 한다.
                //다시 8bit 를 쉬프트 하여 R값을 제거한 G,B 값만 남은 곳에 다시 AND 연산, 255 정규화, 다시 반복해서 RGB 값을 buffer 에 담는다.
            }
        }
        buffer.rewind()
        return buffer
    }

    fun loadLabel(context: Context) {
        // txt 파일 불러오기
        BufferedReader(InputStreamReader(context.assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = ArrayList<String>()
            while (reader.readLine().also { line = it } != null) {
                classList.add(line!!)
            }
            classes = classList.toTypedArray()
        }
    }

    fun loadModel(context: Context) {
        //onnx 파일 불러오기
        val assetManager = context.assets
        val outputFile = File(context.filesDir.toString() + "/" + FILE_NAME)

        assetManager.open(FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
    }

}