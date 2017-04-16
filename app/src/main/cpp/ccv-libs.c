/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
//package top.haoxu13.ohcr;
#include <jni.h>
#include <android/log.h>
#include <ccv.h>

#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, "ccv-libs::", __VA_ARGS__))

JNIEXPORT jobjectArray JNICALL
Java_top_haoxu13_ohcr_TextDetection_swtWordRect(JNIEnv *env, jobject thiz,
                                                jbyteArray array, jint rows, jint cols, jint scanline) {
    // convert to c buffer
    jsize len = (*env)->GetArrayLength (env, array);
    jbyte* buf = (*env)->GetByteArrayElements(env, array, NULL);
    // to ccv image
    ccv_dense_matrix_t* image = 0;
    ccv_read(buf, &image, CCV_IO_GRAY_RAW, (int)rows, (int)cols, (int)scanline);
    // get swt detect words result
    ccv_array_t* result_array = 0;
    result_array = ccv_swt_detect_words(image, ccv_swt_default_params);
    // convert to point array
    ccv_comp_t* result_comps = (ccv_comp_t*)malloc(sizeof(ccv_comp_t)*result_array->size);
    // find Rect Class and construct Rect Array
    jobjectArray rect_array;
    jclass  cls = (*env)->FindClass(env, "org/opencv/core/Rect");
    jmethodID constructor = (*env)->GetMethodID(env, cls, "<init>", "()V");
    jobject init_object = (*env)->NewObject(env, cls, constructor);
    rect_array = (jobjectArray)(*env)->NewObjectArray(env, result_array->size, cls, init_object);
    // get field of Rect Class, "I" represent for int
    jfieldID xField = (*env)->GetFieldID(env, cls, "x", "I");
    jfieldID yField = (*env)->GetFieldID(env, cls, "y", "I");
    jfieldID widthField = (*env)->GetFieldID(env, cls, "width", "I");
    jfieldID heightField = (*env)->GetFieldID(env, cls, "height", "I");
    // set value
    for(int i = 0; i < result_array->size; i++) {
        result_comps[i] = ((ccv_comp_t*)result_array->data)[i];
        jobject element = (*env)->NewObject(env, cls, constructor);
        (*env)->SetIntField(env, element, xField, (jint)result_comps[i].rect.x);
        (*env)->SetIntField(env, element, yField, (jint)result_comps[i].rect.y);
        (*env)->SetIntField(env, element, widthField, (jint)result_comps[i].rect.width);
        (*env)->SetIntField(env, element, heightField, (jint)result_comps[i].rect.height);
        (*env)->SetObjectArrayElement(env, rect_array, i, element);
    }
    free(result_comps);
    return rect_array;
}

JNIEXPORT jbyteArray JNICALL
Java_top_haoxu13_ohcr_TextDetection_swtImage(JNIEnv *env, jobject thiz,
                                             jbyteArray array, jint rows, jint cols, jint scanline) {
    // convert to c buffer
    jsize len = (*env)->GetArrayLength (env, array);
    jbyte* buf = (*env)->GetByteArrayElements(env, array, NULL);
    // to ccv image
    ccv_dense_matrix_t* input = 0;
    ccv_read(buf, &input, CCV_IO_GRAY_RAW, (int)rows, (int)cols, (int)scanline);    // get swt detect words result
    ccv_dense_matrix_t* output = 0;
    ccv_swt(input, &output, 0, ccv_swt_default_params);
    ccv_matrix_t* swtImg = 0;
    ccv_visualize(output, &swtImg, 0);

    jbyteArray return_array = (*env)->NewByteArray(env, len);
    (*env)->SetByteArrayRegion (env, return_array, 0, len, (jbyte*)(ccv_get_dense_matrix(swtImg)->data.u8));
    return return_array;
}

JNIEXPORT jbyteArray JNICALL
Java_top_haoxu13_ohcr_TextDetection_TestRead(JNIEnv *env, jobject thiz,
                                             jbyteArray array, jint rows, jint cols, jint scanline) {
    // convert to c buffer
    jsize len = (*env)->GetArrayLength (env, array);
    jbyte* buf = (*env)->GetByteArrayElements(env, array, NULL);
    // to ccv image
    ccv_dense_matrix_t* input = 0;
    ccv_read(buf, &input, CCV_IO_GRAY_RAW, (int)rows, (int)cols, (int)scanline);
    // get swt detect words result

    jbyteArray return_array = (*env)->NewByteArray(env, len);
    (*env)->SetByteArrayRegion (env, return_array, 0, len, (jbyte*)input->data.u8);

    return return_array;
}