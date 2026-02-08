package com.example.spatialstreamer

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.net.Socket
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var statusText: TextView
    private lateinit var serverIpInput: EditText
    private lateinit var connectButton: Button

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val networkScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val isStreaming = AtomicBoolean(false)

    private var socket: Socket? = null
    private var outputStream: OutputStream? = null

    private val latestFrame = AtomicReference<ByteArray?>(null)

    companion object {
        private const val TAG = "SpatialStreamer"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        private const val SERVER_PORT = 8888
        private const val JPEG_QUALITY = 60
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        statusText = findViewById(R.id.statusText)
        serverIpInput = findViewById(R.id.serverIpInput)
        connectButton = findViewById(R.id.connectButton)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }

        connectButton.setOnClickListener {
            if (isStreaming.get()) stopStreaming()
            else startStreaming(serverIpInput.text.toString())
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { image ->
                if (isStreaming.get()) {
                    latestFrame.set(yuvToJpeg(image))
                }
                image.close()
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun startStreaming(serverIp: String) {
        if (serverIp.isBlank()) return

        networkScope.launch {
            try {
                socket = Socket(serverIp, SERVER_PORT)
                outputStream = socket!!.getOutputStream()
                isStreaming.set(true)

                runOnUiThread {
                    statusText.text = "Status: Streaming"
                    connectButton.text = "Disconnect"
                }

                streamLoop()

            } catch (e: Exception) {
                Log.e(TAG, "Connection failed", e)
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Connection failed",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private suspend fun streamLoop() {
        val stream = outputStream ?: return

        while (isStreaming.get()) {
            val frame = latestFrame.getAndSet(null)
            if (frame != null) {
                val sizeHeader = ByteBuffer.allocate(4)
                    .putInt(frame.size)
                    .array()

                stream.write(sizeHeader)
                stream.write(frame)
            } else {
                delay(1) // tiny yield to avoid busy spin
            }
        }
    }

    private fun stopStreaming() {
        isStreaming.set(false)

        networkScope.launch {
            try {
                socket?.close()
            } catch (_: Exception) {}
            socket = null
            outputStream = null
        }

        statusText.text = "Status: Disconnected"
        connectButton.text = "Connect"
    }

    private fun yuvToJpeg(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            JPEG_QUALITY,
            out
        )

        return out.toByteArray()
    }

    private fun allPermissionsGranted(): Boolean =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) ==
                    PackageManager.PERMISSION_GRANTED
        }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        cameraExecutor.shutdown()
        networkScope.cancel()
    }
}
