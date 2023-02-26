/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Import webgl flags.
import './flags_webgl';
import { backend_util, buffer, DataStorage, engine, env, kernel_impls, KernelBackend, nextFrame, scalar, tidy, util } from '@tensorflow/tfjs-core';
import { getWebGLContext } from './canvas_util';
import { DecodeMatrixProgram } from './decode_matrix_gpu';
import { DecodeMatrixPackedProgram } from './decode_matrix_packed_gpu';
import { EncodeFloatProgram } from './encode_float_gpu';
import { EncodeFloatPackedProgram } from './encode_float_packed_gpu';
import { EncodeMatrixProgram } from './encode_matrix_gpu';
import { EncodeMatrixPackedProgram } from './encode_matrix_packed_gpu';
import { GPGPUContext } from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import { getUniformLocations } from './gpgpu_math';
import { simpleAbsImplCPU } from './kernel_utils/shared';
import { PackProgram } from './pack_gpu';
import { ReshapePackedProgram } from './reshape_packed_gpu';
import * as tex_util from './tex_util';
import { TextureUsage } from './tex_util';
import { TextureManager } from './texture_manager';
import * as unary_op from './unaryop_gpu';
import { UnaryOpProgram } from './unaryop_gpu';
import { UnaryOpPackedProgram } from './unaryop_packed_gpu';
import { UnpackProgram } from './unpack_gpu';
import * as webgl_util from './webgl_util';
const whereImpl = kernel_impls.whereImpl;
export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;
const binaryCaches = {};
export function getBinaryCache(webGLVersion) {
    if (webGLVersion in binaryCaches) {
        return binaryCaches[webGLVersion];
    }
    binaryCaches[webGLVersion] = {};
    return binaryCaches[webGLVersion];
}
// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = env().getNumber('CPU_HANDOFF_SIZE_THRESHOLD');
// Empirically determined constant used to decide the number of MB on GPU
// before we warn about high memory use. The MB are this constant * screen area
// * dpi / 1024 / 1024.
const BEFORE_PAGING_CONSTANT = 600;
function numMBBeforeWarning() {
    if (env().global.screen == null) {
        return 1024; // 1 GB.
    }
    return (env().global.screen.height * env().global.screen.width *
        window.devicePixelRatio) *
        BEFORE_PAGING_CONSTANT / 1024 / 1024;
}
export class MathBackendWebGL extends KernelBackend {
    constructor(gpuResource) {
        super();
        // Maps data ids that have a pending read operation, to list of subscribers.
        this.pendingRead = new WeakMap();
        // List of data ids that are scheduled for disposal, but are waiting on a
        // pending read operation.
        this.pendingDisposal = new WeakSet();
        // Used to count the number of 'shallow' sliced tensors that point to the
        // same data id.
        this.dataRefCount = new WeakMap();
        this.numBytesInGPU = 0;
        // Accumulated time spent (including blocking) in uploading data to webgl.
        this.uploadWaitMs = 0;
        // Accumulated time spent (including blocking in downloading data from webgl.
        this.downloadWaitMs = 0;
        // record the last manual GL Flush time.
        this.lastGlFlushTime = 0;
        this.warnedAboutMemory = false;
        this.pendingDeletes = 0;
        this.disposed = false;
        if (!env().getBool('HAS_WEBGL')) {
            throw new Error('WebGL is not supported on this device');
        }
        let newGPGPU;
        if (gpuResource != null) {
            if (gpuResource instanceof GPGPUContext) {
                newGPGPU = gpuResource;
            }
            else {
                const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'), gpuResource);
                newGPGPU = new GPGPUContext(gl);
            }
            this.binaryCache = {};
            this.gpgpuCreatedLocally = false;
        }
        else {
            const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'));
            newGPGPU = new GPGPUContext(gl);
            this.binaryCache = getBinaryCache(env().getNumber('WEBGL_VERSION'));
            this.gpgpuCreatedLocally = true;
        }
        this.gpgpu = newGPGPU;
        this.canvas = this.gpgpu.gl.canvas;
        this.textureManager = new TextureManager(this.gpgpu);
        this.numMBBeforeWarning = numMBBeforeWarning();
        this.texData = new DataStorage(this, engine());
    }
    nextDataId() {
        return MathBackendWebGL.nextDataId++;
    }
    numDataIds() {
        return this.texData.numDataIds() - this.pendingDeletes;
    }
    // Writes a new entry to the data store with a WebGL texture, and registers it
    // to the texture manager.
    writeTexture(texture, shape, dtype, texHeight, texWidth, channels) {
        // Temporarily create an tensor info to make the texture compatible with
        // the runWebGLProgram's input.
        const input = this.makeTensorInfo(shape, dtype);
        const inData = this.texData.get(input.dataId);
        // Even though the input texture could be unpacked or dense packed, it is
        // always considered as unpacked for EncodeMatrixProgram.
        inData.isPacked = false;
        // Bind texture to the input tensor.
        inData.texture = { texture, texShape: [texHeight, texWidth] };
        inData.texShape = [texHeight, texWidth];
        const shapeAs3D = webgl_util.getShapeAs3D(shape);
        const program = new EncodeMatrixProgram(shapeAs3D, false /* isByteArray */, channels);
        const output = this.runWebGLProgram(program, [input], dtype, [[texHeight, texWidth]]);
        output.shape = shape;
        // Unbind the texture from the input tensor to avoid the texture being
        // released.
        inData.texture = null;
        this.disposeIntermediateTensorInfo(input);
        return output.dataId;
    }
    write(values, shape, dtype) {
        if (env().getBool('WEBGL_CHECK_NUMERICAL_PROBLEMS') ||
            env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64' && values != null) {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const dataId = { id: this.nextDataId() };
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1 });
        return dataId;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.texData.has(dataId)) {
            const tensorData = this.texData.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    /** Increase refCount of a `TextureData`. */
    incRef(dataId) {
        const texData = this.texData.get(dataId);
        texData.refCount++;
    }
    /** Decrease refCount of a `TextureData`. */
    decRef(dataId) {
        if (this.texData.has(dataId)) {
            const texData = this.texData.get(dataId);
            texData.refCount--;
        }
    }
    move(dataId, values, shape, dtype, refCount) {
        if (env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount });
    }
    disposeIntermediateTensorInfo(tensorInfo) {
        this.disposeData(tensorInfo.dataId);
    }
    readSync(dataId) {
        const texData = this.texData.get(dataId);
        const { values, dtype, complexTensorInfos, slice, shape, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.readSync(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (dtype === 'string') {
            return values;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let result;
        if (dtype === 'complex64') {
            const realValues = this.readSync(complexTensorInfos.real.dataId);
            const imagValues = this.readSync(complexTensorInfos.imag.dataId);
            result = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            result = this.getValuesFromTexture(dataId);
        }
        if (shouldTimeProgram) {
            this.downloadWaitMs += util.now() - start;
        }
        return this.convertAndCacheOnCPU(dataId, result);
    }
    async read(dataId) {
        if (this.pendingRead.has(dataId)) {
            const subscribers = this.pendingRead.get(dataId);
            return new Promise(resolve => subscribers.push(resolve));
        }
        const texData = this.texData.get(dataId);
        const { values, shape, slice, dtype, complexTensorInfos, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.read(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (env().getBool('DEBUG')) {
            // getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') caused a blocking GPU call.
            // For performance reason, only check it for debugging. In production,
            // it doesn't handle this use case anyway, so behavior is not changed.
            if (!env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
                env().getNumber('WEBGL_VERSION') === 2) {
                throw new Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
                    `WEBGL_VERSION=2 not yet supported.`);
            }
        }
        let buffer = null;
        let tmpDownloadTarget;
        if (dtype !== 'complex64' && env().get('WEBGL_BUFFER_SUPPORTED')) {
            // Possibly copy the texture into a buffer before inserting a fence.
            tmpDownloadTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpDownloadTarget.dataId);
            buffer = this.gpgpu.createBufferFromTexture(tmpData.texture.texture, ...tex_util.getDenseTexShape(shape));
        }
        this.pendingRead.set(dataId, []);
        if (dtype !== 'complex64') {
            // Create a fence and wait for it to resolve.
            await this.gpgpu.createAndWaitForFence();
        }
        // Download the values from the GPU.
        let vals;
        if (dtype === 'complex64') {
            const ps = await Promise.all([
                this.read(complexTensorInfos.real.dataId),
                this.read(complexTensorInfos.imag.dataId)
            ]);
            const realValues = ps[0];
            const imagValues = ps[1];
            vals = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else if (buffer == null) {
            vals = this.getValuesFromTexture(dataId);
        }
        else {
            const size = util.sizeFromShape(shape);
            vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
        }
        if (tmpDownloadTarget != null) {
            this.disposeIntermediateTensorInfo(tmpDownloadTarget);
        }
        if (buffer != null) {
            const gl = this.gpgpu.gl;
            webgl_util.callAndCheck(gl, () => gl.deleteBuffer(buffer));
        }
        const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
        const subscribers = this.pendingRead.get(dataId);
        this.pendingRead.delete(dataId);
        // Notify all pending reads.
        subscribers.forEach(resolve => resolve(dTypeVals));
        if (this.pendingDisposal.has(dataId)) {
            this.pendingDisposal.delete(dataId);
            if (this.disposeData(dataId)) {
                engine().removeDataId(dataId, this);
            }
            this.pendingDeletes--;
        }
        return dTypeVals;
    }
    /**
     * Read tensor to a new texture that is densely packed for ease of use.
     * @param dataId The source tensor.
     * @param options
     *     customTexShape: Optional. If set, will use the user defined texture
     *     shape to create the texture.
     */
    readToGPU(dataId, options = {}) {
        const texData = this.texData.get(dataId);
        const { values, shape, slice, dtype, isPacked, texture } = texData;
        if (dtype === 'complex64') {
            throw new Error('Does not support reading texture for complex64 dtype.');
        }
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const gpuResouorce = this.readToGPU(res, options);
            this.disposeIntermediateTensorInfo(res);
            return gpuResouorce;
        }
        if (texture == null) {
            if (values != null) {
                throw new Error('Data is not on GPU but on CPU.');
            }
            else {
                throw new Error('There is no data on GPU or CPU.');
            }
        }
        // Decode the texture so that it is stored densely (using four channels).
        const tmpTarget = this.decode(dataId, options.customTexShape);
        // Make engine track this tensor, so that we can dispose it later.
        const tensorRef = engine().makeTensorFromTensorInfo(tmpTarget);
        const tmpData = this.texData.get(tmpTarget.dataId);
        return Object.assign({ tensorRef }, tmpData.texture);
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                const strings = data.map(d => util.decodeString(d));
                return buffer(t.shape, t.dtype, strings);
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, data);
    }
    checkNumericalProblems(values) {
        if (values == null) {
            return;
        }
        for (let i = 0; i < values.length; i++) {
            const num = values[i];
            if (!webgl_util.canBeRepresented(num)) {
                if (env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
                    throw Error(`The value ${num} cannot be represented with your ` +
                        `current settings. Consider enabling float32 rendering: ` +
                        `'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);
                }
                throw Error(`The value ${num} cannot be represented on this device.`);
            }
        }
    }
    getValuesFromTexture(dataId) {
        const { shape, dtype, isPacked } = this.texData.get(dataId);
        const size = util.sizeFromShape(shape);
        if (env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
            const tmpTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpTarget.dataId);
            const vals = this.gpgpu
                .downloadMatrixFromPackedTexture(tmpData.texture.texture, ...tex_util.getDenseTexShape(shape))
                .subarray(0, size);
            this.disposeIntermediateTensorInfo(tmpTarget);
            return vals;
        }
        const shouldUsePackedProgram = env().getBool('WEBGL_PACK') && isPacked === true;
        const outputShape = shouldUsePackedProgram ? webgl_util.getShapeAs3D(shape) : shape;
        const program = shouldUsePackedProgram ?
            new EncodeFloatPackedProgram(outputShape) :
            new EncodeFloatProgram(outputShape);
        const output = this.runWebGLProgram(program, [{ shape: outputShape, dtype, dataId }], 'float32');
        const tmpData = this.texData.get(output.dataId);
        const vals = this.gpgpu
            .downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture.texture, tmpData.texShape[0], tmpData.texShape[1])
            .subarray(0, size);
        this.disposeIntermediateTensorInfo(output);
        return vals;
    }
    timerAvailable() {
        return env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0;
    }
    time(f) {
        const oldActiveTimers = this.activeTimers;
        const newActiveTimers = [];
        let outerMostTime = false;
        if (this.programTimersStack == null) {
            this.programTimersStack = newActiveTimers;
            outerMostTime = true;
        }
        else {
            this.activeTimers.push(newActiveTimers);
        }
        this.activeTimers = newActiveTimers;
        f();
        // needing to split these up because util.flatten only accepts certain types
        const flattenedActiveTimerQueries = util.flatten(this.activeTimers.map((d) => d.query))
            .filter(d => d != null);
        const flattenedActiveTimerNames = util.flatten(this.activeTimers.map((d) => d.name))
            .filter(d => d != null);
        this.activeTimers = oldActiveTimers;
        if (outerMostTime) {
            this.programTimersStack = null;
        }
        const res = {
            uploadWaitMs: this.uploadWaitMs,
            downloadWaitMs: this.downloadWaitMs,
            kernelMs: null,
            wallMs: null // will be filled by the engine
        };
        return (async () => {
            if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') >
                0) {
                const kernelMs = await Promise.all(flattenedActiveTimerQueries);
                res['kernelMs'] = util.sum(kernelMs);
                res['getExtraProfileInfo'] = () => kernelMs
                    .map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
                    .map(d => `${d.name}: ${d.ms}`)
                    .join(', ');
            }
            else {
                res['kernelMs'] = {
                    error: 'WebGL query timers are not supported in this environment.'
                };
            }
            this.uploadWaitMs = 0;
            this.downloadWaitMs = 0;
            return res;
        })();
    }
    memory() {
        return {
            unreliable: false,
            numBytesInGPU: this.numBytesInGPU,
            numBytesInGPUAllocated: this.textureManager.numBytesAllocated,
            numBytesInGPUFree: this.textureManager.numBytesFree
        };
    }
    startTimer() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.beginQuery();
        }
        return { startMs: util.now(), endMs: null };
    }
    endTimer(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            this.gpgpu.endQuery();
            return query;
        }
        query.endMs = util.now();
        return query;
    }
    async getQueryTime(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.waitForQueryAndGetTime(query);
        }
        const timerQuery = query;
        return timerQuery.endMs - timerQuery.startMs;
    }
    /**
     * Decrease the RefCount on the dataId and dispose the memory if the dataId
     * has 0 refCount. If there are pending read on the data, the disposal would
     * added to the pending delete queue. Return true if the dataId is removed
     * from backend or the backend does not contain the dataId, false if the
     * dataId is not removed. Memory may or may not be released even when dataId
     * is removed, which also depends on dataRefCount, see `releaseGPU`.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.pendingDisposal.has(dataId)) {
            return false;
        }
        // No-op if already disposed.
        if (!this.texData.has(dataId)) {
            return true;
        }
        // if force flag is set, change refCount to 0, this would ensure disposal
        // when added to the pendingDisposal queue. Memory may or may not be
        // released, which also depends on dataRefCount, see `releaseGPU`.
        if (force) {
            this.texData.get(dataId).refCount = 0;
        }
        else {
            this.texData.get(dataId).refCount--;
        }
        if (!force && this.texData.get(dataId).refCount > 0) {
            return false;
        }
        if (this.pendingRead.has(dataId)) {
            this.pendingDisposal.add(dataId);
            this.pendingDeletes++;
            return false;
        }
        this.releaseGPUData(dataId);
        const { complexTensorInfos } = this.texData.get(dataId);
        if (complexTensorInfos != null) {
            this.disposeData(complexTensorInfos.real.dataId, force);
            this.disposeData(complexTensorInfos.imag.dataId, force);
        }
        this.texData.delete(dataId);
        return true;
    }
    releaseGPUData(dataId) {
        const { texture, dtype, texShape, usage, isPacked, slice } = this.texData.get(dataId);
        const key = slice && slice.origDataId || dataId;
        const refCount = this.dataRefCount.get(key);
        if (refCount > 1) {
            this.dataRefCount.set(key, refCount - 1);
        }
        else {
            this.dataRefCount.delete(key);
            if (texture != null) {
                this.numBytesInGPU -= this.computeBytes(texShape, dtype);
                this.textureManager.releaseTexture(texture, texShape, usage, isPacked);
            }
        }
        const texData = this.texData.get(dataId);
        texData.texture = null;
        texData.texShape = null;
        texData.isPacked = false;
        texData.slice = null;
    }
    getTexture(dataId) {
        this.uploadToGPU(dataId);
        return this.texData.get(dataId).texture.texture;
    }
    /**
     * Returns internal information for the specific data bucket. Used in unit
     * tests.
     */
    getDataInfo(dataId) {
        return this.texData.get(dataId);
    }
    /*
    Tests whether all the inputs to an op are small and on the CPU. This heuristic
    determines when it would be faster to execute a kernel on the CPU. WebGL
    kernels opt into running this check and forwarding when appropriate.
    TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
    sustainable strategy for optimizing backend execution of ops.
     */
    shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
        return env().getBool('WEBGL_CPU_FORWARD') &&
            inputs.every(input => this.texData.get(input.dataId).texture == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
    }
    getGPGPUContext() {
        return this.gpgpu;
    }
    where(condition) {
        backend_util.warn('tf.where() in webgl locks the UI thread. ' +
            'Call tf.whereAsync() instead');
        const condVals = condition.dataSync();
        return whereImpl(condition.shape, condVals);
    }
    packedUnaryOp(x, op, dtype) {
        const program = new UnaryOpPackedProgram(x.shape, op);
        const outInfo = this.compileAndRun(program, [x], dtype);
        return engine().makeTensorFromTensorInfo(outInfo);
    }
    // TODO(msoulanille) remove this once the backend has been modularized
    // a copy is needed here to break a circular dependency.
    // Also remove the op from unary_op.
    abs(x) {
        // TODO: handle cases when x is complex.
        if (this.shouldExecuteOnCPU([x]) && x.dtype !== 'complex64') {
            const outValues = simpleAbsImplCPU(this.texData.get(x.dataId).values);
            return this.makeOutput(x.shape, x.dtype, outValues);
        }
        if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
            return this.packedUnaryOp(x, unary_op.ABS, x.dtype);
        }
        const program = new UnaryOpProgram(x.shape, unary_op.ABS);
        const outInfo = this.compileAndRun(program, [x]);
        return engine().makeTensorFromTensorInfo(outInfo);
    }
    makeTensorInfo(shape, dtype, values) {
        let dataId;
        if (dtype === 'string' && values != null && values.length > 0 &&
            util.isString(values[0])) {
            const encodedValues = values.map(d => util.encodeString(d));
            dataId = this.write(encodedValues, shape, dtype);
        }
        else {
            dataId = this.write(values, shape, dtype);
        }
        this.texData.get(dataId).usage = null;
        return { dataId, shape, dtype };
    }
    makeOutput(shape, dtype, values) {
        return engine().makeTensorFromTensorInfo(this.makeTensorInfo(shape, dtype, values), this);
    }
    unpackTensor(input) {
        const program = new UnpackProgram(input.shape);
        return this.runWebGLProgram(program, [input], input.dtype);
    }
    packTensor(input) {
        const program = new PackProgram(input.shape);
        const preventEagerUnpackingOutput = true;
        return this.runWebGLProgram(program, [input], input.dtype, null /* customUniformValues */, preventEagerUnpackingOutput);
    }
    packedReshape(input, afterShape) {
        const input3DShape = [
            webgl_util.getBatchDim(input.shape),
            ...webgl_util.getRowsCols(input.shape)
        ];
        const input3D = {
            dtype: input.dtype,
            shape: input3DShape,
            dataId: input.dataId
        };
        const afterShapeAs3D = [
            webgl_util.getBatchDim(afterShape), ...webgl_util.getRowsCols(afterShape)
        ];
        const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
        const preventEagerUnpackingOfOutput = true;
        const customValues = [input3DShape];
        const output = this.runWebGLProgram(program, [input3D], input.dtype, customValues, preventEagerUnpackingOfOutput);
        return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
    }
    decode(dataId, customTexShape) {
        const texData = this.texData.get(dataId);
        const { isPacked, shape, dtype } = texData;
        if (customTexShape != null) {
            const size = util.sizeFromShape(shape);
            const texSize = customTexShape[0] * customTexShape[1] * 4;
            util.assert(size <= texSize, () => 'customTexShape is too small. ' +
                'Row * Column * 4 should be equal or larger than the ' +
                'size of the tensor data.');
        }
        const shapeAs3D = webgl_util.getShapeAs3D(shape);
        let program;
        if (isPacked) {
            program = new DecodeMatrixPackedProgram(shapeAs3D);
        }
        else {
            program = new DecodeMatrixProgram(shapeAs3D);
        }
        const preventEagerUnpackingOfOutput = true;
        const customValues = [customTexShape != null ? customTexShape :
                tex_util.getDenseTexShape(shapeAs3D)];
        const out = this.runWebGLProgram(program, [{ shape: shapeAs3D, dtype, dataId }], dtype, customValues, preventEagerUnpackingOfOutput, customTexShape);
        return { dtype, shape, dataId: out.dataId };
    }
    runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false, customTexShape) {
        const output = this.makeTensorInfo(program.outputShape, outputDtype);
        const outData = this.texData.get(output.dataId);
        if (program.packedOutput) {
            outData.isPacked = true;
        }
        if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
            const texelShape = customTexShape != null ?
                customTexShape :
                tex_util.getDenseTexShape(program.outputShape);
            // For a densely packed output, we explicitly set texShape
            // so it doesn't get assigned later according to our typical packing
            // scheme wherein a single texel can only contain values from adjacent
            // rows/cols.
            outData.texShape = texelShape.map(d => d * 2);
        }
        if (program.outTexUsage != null) {
            outData.usage = program.outTexUsage;
        }
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            outData.values =
                util.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        const dataToDispose = [];
        const inputsData = inputs.map(input => {
            if (input.dtype === 'complex64') {
                throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                    `dtypes, please separate the program into real and imaginary ` +
                    `parts.`);
            }
            let texData = this.texData.get(input.dataId);
            if (texData.texture == null) {
                if (!program.packedInputs &&
                    util.sizeFromShape(input.shape) <=
                        env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
                    // Upload small tensors that live on the CPU as uniforms, not as
                    // textures. Do this only when the environment supports 32bit floats
                    // due to problems when comparing 16bit floats with 32bit floats.
                    // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
                    // possible for packed shaders to sample from uniforms.
                    return {
                        shape: input.shape,
                        texData: null,
                        isUniform: true,
                        uniformValues: texData.values
                    };
                }
                // This ensures that if a packed program's inputs have not yet been
                // uploaded to the GPU, they get uploaded as packed right off the bat.
                if (program.packedInputs) {
                    texData.isPacked = true;
                    texData.shape = input.shape;
                }
            }
            this.uploadToGPU(input.dataId);
            if (!!texData.isPacked !== !!program.packedInputs) {
                input = texData.isPacked ? this.unpackTensor(input) :
                    this.packTensor(input);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
            }
            else if (texData.isPacked &&
                !webgl_util.isReshapeFree(texData.shape, input.shape)) {
                // This is a special case where a texture exists for a tensor
                // but the shapes are incompatible (due to packing constraints) because
                // the tensor did not have a chance to go through the packed reshape
                // shader. This only happens when we reshape the *same* tensor to form
                // *distinct* inputs to an op, e.g. dotting a vector with itself. This
                // case will disappear once packed uploading is the default.
                const savedInput = input;
                const targetShape = input.shape;
                input.shape = texData.shape;
                input = this.packedReshape(input, targetShape);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
                savedInput.shape = targetShape;
            }
            return { shape: input.shape, texData, isUniform: false };
        });
        this.uploadToGPU(output.dataId);
        const outputData = { shape: output.shape, texData: outData, isUniform: false };
        const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
        const binary = this.getAndSaveBinary(key, () => {
            return gpgpu_math.compileProgram(this.gpgpu, program, inputsData, outputData);
        });
        const shouldTimeProgram = this.activeTimers != null;
        let query;
        if (shouldTimeProgram) {
            query = this.startTimer();
        }
        if (!env().get('ENGINE_COMPILE_ONLY')) {
            gpgpu_math.runProgram(this.gpgpu, binary, inputsData, outputData, customUniformValues);
        }
        dataToDispose.forEach(info => this.disposeIntermediateTensorInfo(info));
        if (shouldTimeProgram) {
            query = this.endTimer(query);
            this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
        }
        const glFlushThreshold = env().get('WEBGL_FLUSH_THRESHOLD');
        // Manually GL flush requested
        if (glFlushThreshold > 0) {
            const time = util.now();
            if ((time - this.lastGlFlushTime) > glFlushThreshold) {
                this.gpgpu.gl.flush();
                this.lastGlFlushTime = time;
            }
        }
        if (!env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
            preventEagerUnpackingOfOutput === false) {
            const unpacked = this.unpackTensor(output);
            this.disposeIntermediateTensorInfo(output);
            return unpacked;
        }
        return output;
    }
    compileAndRun(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false) {
        outputDtype = outputDtype || inputs[0].dtype;
        const outInfo = this.runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput);
        return outInfo;
    }
    getAndSaveBinary(key, getBinary) {
        if (!(key in this.binaryCache)) {
            this.binaryCache[key] = getBinary();
        }
        return this.binaryCache[key];
    }
    getTextureManager() {
        return this.textureManager;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        // Avoid disposing the compiled webgl programs during unit testing because
        // it slows down test execution.
        if (!env().getBool('IS_TEST')) {
            const allKeys = Object.keys(this.binaryCache);
            allKeys.forEach(key => {
                this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
                delete this.binaryCache[key];
            });
        }
        this.textureManager.dispose();
        if (this.canvas != null &&
            (typeof (HTMLCanvasElement) !== 'undefined' &&
                this.canvas instanceof HTMLCanvasElement)) {
            this.canvas.remove();
        }
        else {
            this.canvas = null;
        }
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.program = null;
            this.gpgpu.dispose();
        }
        this.disposed = true;
    }
    floatPrecision() {
        if (this.floatPrecisionValue == null) {
            this.floatPrecisionValue = tidy(() => {
                if (!env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
                    // Momentarily switching DEBUG flag to false so we don't throw an
                    // error trying to upload a small value.
                    const debugFlag = env().getBool('DEBUG');
                    env().set('DEBUG', false);
                    const underflowCheckValue = this.abs(scalar(1e-8)).dataSync()[0];
                    env().set('DEBUG', debugFlag);
                    if (underflowCheckValue > 0) {
                        return 32;
                    }
                }
                return 16;
            });
        }
        return this.floatPrecisionValue;
    }
    /** Returns the smallest representable number.  */
    epsilon() {
        return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
    }
    uploadToGPU(dataId) {
        const texData = this.texData.get(dataId);
        const { shape, dtype, values, texture, usage, isPacked } = texData;
        if (texture != null) {
            // Array is already on GPU. No-op.
            return;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let texShape = texData.texShape;
        if (texShape == null) {
            // This texShape may not be the final texture shape. For packed or dense
            // textures, the texShape will be changed when textures are created.
            texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
            texData.texShape = texShape;
        }
        if (values != null) {
            const shapeAs3D = webgl_util.getShapeAs3D(shape);
            let program;
            let width = texShape[1], height = texShape[0];
            const isByteArray = values instanceof Uint8Array || values instanceof Uint8ClampedArray;
            // texture for float array is PhysicalTextureType.PACKED_2X2_FLOAT32, we
            // need to make sure the upload uses the same packed size
            if (isPacked || !isByteArray) {
                [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(texShape[0], texShape[1]);
            }
            if (isPacked) {
                program = new EncodeMatrixPackedProgram(shapeAs3D, isByteArray);
            }
            else {
                program = new EncodeMatrixProgram(shapeAs3D, isByteArray);
            }
            // TexShape for float array needs to be the original shape, which byte
            // array needs to be packed size. This allow the data upload shape to be
            // matched with texture creation logic.
            const tempDenseInputTexShape = isByteArray ? [height, width] : texShape;
            const tempDenseInputHandle = this.makeTensorInfo(tempDenseInputTexShape, dtype);
            const tempDenseInputTexData = this.texData.get(tempDenseInputHandle.dataId);
            if (isByteArray) {
                tempDenseInputTexData.usage = TextureUsage.PIXELS;
            }
            else {
                tempDenseInputTexData.usage = TextureUsage.UPLOAD;
            }
            tempDenseInputTexData.texShape = tempDenseInputTexShape;
            this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(tempDenseInputHandle.dataId), width, height, values);
            const customValues = [[height, width]];
            // We want the output to remain packed regardless of the value of
            // WEBGL_PACK.
            const preventEagerUnpacking = true;
            const encodedOutputTarget = this.runWebGLProgram(program, [tempDenseInputHandle], dtype, customValues, preventEagerUnpacking);
            // Have the original texture assume the identity of the encoded output.
            const outputTexData = this.texData.get(encodedOutputTarget.dataId);
            texData.texShape = outputTexData.texShape;
            texData.isPacked = outputTexData.isPacked;
            texData.usage = outputTexData.usage;
            if (!env().get('ENGINE_COMPILE_ONLY')) {
                texData.texture = outputTexData.texture;
                // Once uploaded, don't store the values on cpu.
                texData.values = null;
                this.texData.delete(encodedOutputTarget.dataId);
            }
            else {
                this.disposeData(encodedOutputTarget.dataId);
            }
            this.disposeIntermediateTensorInfo(tempDenseInputHandle);
            if (shouldTimeProgram) {
                this.uploadWaitMs += util.now() - start;
            }
        }
        else {
            const newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
            texData.texture = newTexture;
        }
    }
    convertAndCacheOnCPU(dataId, float32Values) {
        const texData = this.texData.get(dataId);
        const { dtype } = texData;
        if (float32Values != null) {
            texData.values = float32ToTypedArray(float32Values, dtype);
        }
        return texData.values;
    }
    acquireTexture(texShape, texType, dtype, isPacked) {
        this.numBytesInGPU += this.computeBytes(texShape, dtype);
        if (!this.warnedAboutMemory &&
            this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
            const mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
            this.warnedAboutMemory = true;
            console.warn(`High memory usage in GPU: ${mb} MB, ` +
                `most likely due to a memory leak`);
        }
        return this.textureManager.acquireTexture(texShape, texType, isPacked);
    }
    computeBytes(shape, dtype) {
        return shape[0] * shape[1] * util.bytesPerElement(dtype);
    }
    checkCompileCompletion() {
        for (const [, binary] of Object.entries(this.binaryCache)) {
            this.checkCompletion_(binary);
        }
    }
    async checkCompileCompletionAsync() {
        const ps = [];
        if (this.gpgpu.parallelCompilationExtension) {
            for (const [, binary] of Object.entries(this.binaryCache)) {
                ps.push(this.checkCompletionAsync_(binary));
            }
            return Promise.all(ps);
        }
        else {
            for (const [, binary] of Object.entries(this.binaryCache)) {
                const p = new Promise((resolve) => {
                    try {
                        this.checkCompletion_(binary);
                        resolve(true);
                    }
                    catch (error) {
                        throw error;
                    }
                });
                ps.push(p);
            }
            return Promise.all(ps);
        }
    }
    async checkCompletionAsync_(binary) {
        if (this.gpgpu.gl.getProgramParameter(binary.webGLProgram, this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)) {
            return this.checkCompletion_(binary);
        }
        else {
            await nextFrame();
            return this.checkCompletionAsync_(binary);
        }
    }
    checkCompletion_(binary) {
        if (this.gpgpu.gl.getProgramParameter(binary.webGLProgram, this.gpgpu.gl.LINK_STATUS) === false) {
            console.log(this.gpgpu.gl.getProgramInfoLog(binary.webGLProgram));
            if (this.gpgpu.gl.getShaderParameter(binary.fragmentShader, this.gpgpu.gl.COMPILE_STATUS) === false) {
                webgl_util.logShaderSourceAndInfoLog(binary.source, this.gpgpu.gl.getShaderInfoLog(binary.fragmentShader));
                throw new Error('Failed to compile fragment shader.');
            }
            throw new Error('Failed to link vertex and fragment shaders.');
        }
        return true;
    }
    getUniformLocations() {
        for (const [, binary] of Object.entries(this.binaryCache)) {
            const { uniformLocations, customUniformLocations, infLoc, nanLoc, inShapesLocations, inTexShapesLocations, outShapeLocation, outShapeStridesLocation, outTexShapeLocation } = getUniformLocations(this.gpgpu, binary.program, binary.webGLProgram);
            binary.uniformLocations = uniformLocations;
            binary.customUniformLocations = customUniformLocations;
            binary.infLoc = infLoc;
            binary.nanLoc = nanLoc;
            binary.inShapesLocations = inShapesLocations;
            binary.inTexShapesLocations = inTexShapesLocations;
            binary.outShapeLocation = outShapeLocation;
            binary.outShapeStridesLocation = outShapeStridesLocation;
            binary.outTexShapeLocation = outTexShapeLocation;
        }
    }
    /**
     * Create a TF.js tensor out of an existing WebGL texture. A new texture will
     * be created.
     */
    createTensorFromGPUData(values, shape, dtype) {
        values.channels = values.channels || 'RGBA';
        const { texture, height, width, channels } = values;
        const backend = engine().backend;
        // Have to throw an error, otherwise WebGL just warns and returns wrong
        // values.
        if (!backend.gpgpu.gl.isTexture(texture)) {
            throw new Error(`The texture is invalid. Also, please make sure the texture and ` +
                `the TFJS WebGL backend are using the same canvas. If you want to ` +
                `use your own custom canvas, you have to create and use the custom ` +
                `TFJS WebGL backend created from the canvas through ` +
                `'new tf.MathBackendWebGL(customCanvas)'.`);
        }
        const dataId = backend.writeTexture(texture, shape, dtype, height, width, channels);
        return engine().makeTensorFromDataId(dataId, shape, dtype, backend);
    }
}
MathBackendWebGL.nextDataId = 0;
function float32ToTypedArray(a, dtype) {
    if (dtype === 'float32' || dtype === 'complex64') {
        return a;
    }
    else if (dtype === 'int32' || dtype === 'bool') {
        const result = (dtype === 'int32') ? new Int32Array(a.length) :
            new Uint8Array(a.length);
        for (let i = 0; i < result.length; ++i) {
            result[i] = Math.round(a[i]);
        }
        return result;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93ZWJnbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvYmFja2VuZF93ZWJnbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxzQkFBc0I7QUFDdEIsT0FBTyxlQUFlLENBQUM7QUFHdkIsT0FBTyxFQUFDLFlBQVksRUFBaUIsTUFBTSxFQUFVLFdBQVcsRUFBa0MsTUFBTSxFQUFFLEdBQUcsRUFBVyxZQUFZLEVBQUUsYUFBYSxFQUFjLFNBQVMsRUFBeUMsTUFBTSxFQUF3RCxJQUFJLEVBQTBCLElBQUksRUFBWSxNQUFNLHVCQUF1QixDQUFDO0FBQzdWLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDOUMsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDeEQsT0FBTyxFQUFDLHlCQUF5QixFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDckUsT0FBTyxFQUFDLGtCQUFrQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDdEQsT0FBTyxFQUFDLHdCQUF3QixFQUFDLE1BQU0sMkJBQTJCLENBQUM7QUFDbkUsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDeEQsT0FBTyxFQUFDLHlCQUF5QixFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDckUsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzdDLE9BQU8sS0FBSyxVQUFVLE1BQU0sY0FBYyxDQUFDO0FBQzNDLE9BQU8sRUFBQyxtQkFBbUIsRUFBd0MsTUFBTSxjQUFjLENBQUM7QUFDeEYsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFDdkQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUN2QyxPQUFPLEVBQUMsb0JBQW9CLEVBQUMs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Q29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8vIEltcG9ydCB3ZWJnbCBmbGFncy5cbmltcG9ydCAnLi9mbGFnc193ZWJnbCc7XG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgQmFja2VuZFZhbHVlcywgYnVmZmVyLCBEYXRhSWQsIERhdGFTdG9yYWdlLCBEYXRhVG9HUFVXZWJHTE9wdGlvbiwgRGF0YVR5cGUsIGVuZ2luZSwgZW52LCBHUFVEYXRhLCBrZXJuZWxfaW1wbHMsIEtlcm5lbEJhY2tlbmQsIE1lbW9yeUluZm8sIG5leHRGcmFtZSwgTnVtZXJpY0RhdGFUeXBlLCBSYW5rLCBSZWN1cnNpdmVBcnJheSwgc2NhbGFyLCBTaGFwZU1hcCwgVGVuc29yLCBUZW5zb3IyRCwgVGVuc29yQnVmZmVyLCBUZW5zb3JJbmZvLCB0aWR5LCBUaW1pbmdJbmZvLCBUeXBlZEFycmF5LCB1dGlsLCBXZWJHTERhdGF9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldFdlYkdMQ29udGV4dH0gZnJvbSAnLi9jYW52YXNfdXRpbCc7XG5pbXBvcnQge0RlY29kZU1hdHJpeFByb2dyYW19IGZyb20gJy4vZGVjb2RlX21hdHJpeF9ncHUnO1xuaW1wb3J0IHtEZWNvZGVNYXRyaXhQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2RlY29kZV9tYXRyaXhfcGFja2VkX2dwdSc7XG5pbXBvcnQge0VuY29kZUZsb2F0UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfZmxvYXRfZ3B1JztcbmltcG9ydCB7RW5jb2RlRmxvYXRQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2VuY29kZV9mbG9hdF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7RW5jb2RlTWF0cml4UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfbWF0cml4X2dwdSc7XG5pbXBvcnQge0VuY29kZU1hdHJpeFBhY2tlZFByb2dyYW19IGZyb20gJy4vZW5jb2RlX21hdHJpeF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtnZXRVbmlmb3JtTG9jYXRpb25zLCBHUEdQVUJpbmFyeSwgR1BHUFVQcm9ncmFtLCBUZW5zb3JEYXRhfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtzaW1wbGVBYnNJbXBsQ1BVfSBmcm9tICcuL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuaW1wb3J0IHtQYWNrUHJvZ3JhbX0gZnJvbSAnLi9wYWNrX2dwdSc7XG5pbXBvcnQge1Jlc2hhcGVQYWNrZWRQcm9ncmFtfSBmcm9tICcuL3Jlc2hhcGVfcGFja2VkX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbCBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7VGV4dHVyZSwgVGV4dHVyZURhdGEsIFRleHR1cmVVc2FnZX0gZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB1bmFyeV9vcCBmcm9tICcuL3VuYXJ5b3BfZ3B1JztcbmltcG9ydCB7VW5hcnlPcFByb2dyYW19IGZyb20gJy4vdW5hcnlvcF9ncHUnO1xuaW1wb3J0IHtVbmFyeU9wUGFja2VkUHJvZ3JhbX0gZnJvbSAnLi91bmFyeW9wX3BhY2tlZF9ncHUnO1xuaW1wb3J0IHtVbnBhY2tQcm9ncmFtfSBmcm9tICcuL3VucGFja19ncHUnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5jb25zdCB3aGVyZUltcGwgPSBrZXJuZWxfaW1wbHMud2hlcmVJbXBsO1xuXG5leHBvcnQgY29uc3QgRVBTSUxPTl9GTE9BVDMyID0gMWUtNztcbmV4cG9ydCBjb25zdCBFUFNJTE9OX0ZMT0FUMTYgPSAxZS00O1xuXG50eXBlIEtlcm5lbEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZzsgcXVlcnk6IFByb21pc2U8bnVtYmVyPjtcbn07XG5cbmV4cG9ydCB0eXBlIFRpbWVyTm9kZSA9IFJlY3Vyc2l2ZUFycmF5PEtlcm5lbEluZm8+fEtlcm5lbEluZm87XG5leHBvcnQgaW50ZXJmYWNlIENQVVRpbWVyUXVlcnkge1xuICBzdGFydE1zOiBudW1iZXI7XG4gIGVuZE1zPzogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdMTWVtb3J5SW5mbyBleHRlbmRzIE1lbW9yeUluZm8ge1xuICBudW1CeXRlc0luR1BVOiBudW1iZXI7XG4gIC8vIFRyYWNrcyB0aGUgdG90YWwgbnVtYmVyIG9mIGJ5dGVzIGFsbG9jYXRlZCBvbiB0aGUgR1BVLCBhY2NvdW50aW5nIGZvciB0aGVcbiAgLy8gcGh5c2ljYWwgdGV4dHVyZSB0eXBlLlxuICBudW1CeXRlc0luR1BVQWxsb2NhdGVkOiBudW1iZXI7XG4gIC8vIFRyYWNrcyBieXRlIHNpemUgb2YgdGV4dHVyZXMgdGhhdCB3ZXJlIGNyZWF0ZWQgYW5kIHRoZW4gbWFkZSBhdmFpbGFibGUgZm9yXG4gIC8vIHJldXNlIChkaXNwb3NlZCkuXG4gIG51bUJ5dGVzSW5HUFVGcmVlOiBudW1iZXI7XG4gIHVucmVsaWFibGU6IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xUaW1pbmdJbmZvIGV4dGVuZHMgVGltaW5nSW5mbyB7XG4gIHVwbG9hZFdhaXRNczogbnVtYmVyO1xuICBkb3dubG9hZFdhaXRNczogbnVtYmVyO1xufVxuXG5jb25zdCBiaW5hcnlDYWNoZXM6IHtbd2ViR0xWZXJzaW9uOiBzdHJpbmddOiB7W2tleTogc3RyaW5nXTogR1BHUFVCaW5hcnl9fSA9IHt9O1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0QmluYXJ5Q2FjaGUod2ViR0xWZXJzaW9uOiBudW1iZXIpIHtcbiAgaWYgKHdlYkdMVmVyc2lvbiBpbiBiaW5hcnlDYWNoZXMpIHtcbiAgICByZXR1cm4gYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl07XG4gIH1cbiAgYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl0gPSB7fTtcbiAgcmV0dXJuIGJpbmFyeUNhY2hlc1t3ZWJHTFZlcnNpb25dO1xufVxuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGV0ZXJtaW5lIHNpemUgdGhyZXNob2xkIGZvciBoYW5kaW5nXG4vLyBvZmYgZXhlY3V0aW9uIHRvIHRoZSBDUFUuXG5jb25zdCBDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCA9XG4gICAgZW52KCkuZ2V0TnVtYmVyKCdDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCcpO1xuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGVjaWRlIHRoZSBudW1iZXIgb2YgTUIgb24gR1BVXG4vLyBiZWZvcmUgd2Ugd2FybiBhYm91dCBoaWdoIG1lbW9yeSB1c2UuIFRoZSBNQiBhcmUgdGhpcyBjb25zdGFudCAqIHNjcmVlbiBhcmVhXG4vLyAqIGRwaSAvIDEwMjQgLyAxMDI0LlxuY29uc3QgQkVGT1JFX1BBR0lOR19DT05TVEFOVCA9IDYwMDtcbmZ1bmN0aW9uIG51bU1CQmVmb3JlV2FybmluZygpOiBudW1iZXIge1xuICBpZiAoZW52KCkuZ2xvYmFsLnNjcmVlbiA9PSBudWxsKSB7XG4gICAgcmV0dXJuIDEwMjQ7ICAvLyAxIEdCLlxuICB9XG4gIHJldHVybiAoZW52KCkuZ2xvYmFsLnNjcmVlbi5oZWlnaHQgKiBlbnYoKS5nbG9iYWwuc2NyZWVuLndpZHRoICpcbiAgICAgICAgICB3aW5kb3cuZGV2aWNlUGl4ZWxSYXRpbykgKlxuICAgICAgQkVGT1JFX1BBR0lOR19DT05TVEFOVCAvIDEwMjQgLyAxMDI0O1xufVxuXG5leHBvcnQgY2xhc3MgTWF0aEJhY2tlbmRXZWJHTCBleHRlbmRzIEtlcm5lbEJhY2tlbmQge1xuICB0ZXhEYXRhOiBEYXRhU3RvcmFnZTxUZXh0dXJlRGF0YT47XG4gIGdwZ3B1OiBHUEdQVUNvbnRleHQ7XG5cbiAgcHJpdmF0ZSBzdGF0aWMgbmV4dERhdGFJZCA9IDA7XG4gIHByaXZhdGUgbmV4dERhdGFJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBNYXRoQmFja2VuZFdlYkdMLm5leHREYXRhSWQrKztcbiAgfVxuICAvLyBNYXBzIGRhdGEgaWRzIHRoYXQgaGF2ZSBhIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24sIHRvIGxpc3Qgb2Ygc3Vic2NyaWJlcnMuXG4gIHByaXZhdGUgcGVuZGluZ1JlYWQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIEFycmF5PChhcnI6IFR5cGVkQXJyYXkpID0+IHZvaWQ+PigpO1xuICAvLyBMaXN0IG9mIGRhdGEgaWRzIHRoYXQgYXJlIHNjaGVkdWxlZCBmb3IgZGlzcG9zYWwsIGJ1dCBhcmUgd2FpdGluZyBvbiBhXG4gIC8vIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24uXG4gIHByaXZhdGUgcGVuZGluZ0Rpc3Bvc2FsID0gbmV3IFdlYWtTZXQ8RGF0YUlkPigpO1xuXG4gIC8vIFVzZWQgdG8gY291bnQgdGhlIG51bWJlciBvZiAnc2hhbGxvdycgc2xpY2VkIHRlbnNvcnMgdGhhdCBwb2ludCB0byB0aGVcbiAgLy8gc2FtZSBkYXRhIGlkLlxuICBkYXRhUmVmQ291bnQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIG51bWJlcj4oKTtcbiAgcHJpdmF0ZSBudW1CeXRlc0luR1BVID0gMDtcblxuICBwcml2YXRlIGNhbnZhczogSFRNTENhbnZhc0VsZW1lbnR8T2Zmc2NyZWVuQ2FudmFzO1xuXG4gIHByaXZhdGUgcHJvZ3JhbVRpbWVyc1N0YWNrOiBUaW1lck5vZGVbXTtcbiAgcHJpdmF0ZSBhY3RpdmVUaW1lcnM6IFRpbWVyTm9kZVtdO1xuICAvLyBBY2N1bXVsYXRlZCB0aW1lIHNwZW50IChpbmNsdWRpbmcgYmxvY2tpbmcpIGluIHVwbG9hZGluZyBkYXRhIHRvIHdlYmdsLlxuICBwcml2YXRlIHVwbG9hZFdhaXRNcyA9IDA7XG4gIC8vIEFjY3VtdWxhdGVkIHRpbWUgc3BlbnQgKGluY2x1ZGluZyBibG9ja2luZyBpbiBkb3dubG9hZGluZyBkYXRhIGZyb20gd2ViZ2wuXG4gIHByaXZhdGUgZG93bmxvYWRXYWl0TXMgPSAwO1xuXG4gIC8vIHJlY29yZCB0aGUgbGFzdCBtYW51YWwgR0wgRmx1c2ggdGltZS5cbiAgcHJpdmF0ZSBsYXN0R2xGbHVzaFRpbWUgPSAwO1xuXG4gIC8vIE51bWJlciBvZiBiaXRzIG9mIHByZWNpc2lvbiBvZiB0aGlzIGJhY2tlbmQuXG4gIHByaXZhdGUgZmxvYXRQcmVjaXNpb25WYWx1ZTogMzJ8MTY7XG5cbiAgcHJpdmF0ZSB0ZXh0dXJlTWFuYWdlcjogVGV4dHVyZU1hbmFnZXI7XG4gIHByaXZhdGUgYmluYXJ5Q2FjaGU6IHtba2V5OiBzdHJpbmddOiBHUEdQVUJpbmFyeX07XG4gIHByaXZhdGUgZ3BncHVDcmVhdGVkTG9jYWxseTogYm9vbGVhbjtcbiAgcHJpdmF0ZSBudW1NQkJlZm9yZVdhcm5pbmc6IG51bWJlcjtcbiAgcHJpdmF0ZSB3YXJuZWRBYm91dE1lbW9yeSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGdwdVJlc291cmNlPzogR1BHUFVDb250ZXh0fEhUTUxDYW52YXNFbGVtZW50fE9mZnNjcmVlbkNhbnZhcykge1xuICAgIHN1cGVyKCk7XG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdIQVNfV0VCR0wnKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdXZWJHTCBpcyBub3Qgc3VwcG9ydGVkIG9uIHRoaXMgZGV2aWNlJyk7XG4gICAgfVxuXG4gICAgbGV0IG5ld0dQR1BVO1xuICAgIGlmIChncHVSZXNvdXJjZSAhPSBudWxsKSB7XG4gICAgICBpZiAoZ3B1UmVzb3VyY2UgaW5zdGFuY2VvZiBHUEdQVUNvbnRleHQpIHtcbiAgICAgICAgbmV3R1BHUFUgPSBncHVSZXNvdXJjZTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IGdsID1cbiAgICAgICAgICAgIGdldFdlYkdMQ29udGV4dChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSwgZ3B1UmVzb3VyY2UpO1xuICAgICAgICBuZXdHUEdQVSA9IG5ldyBHUEdQVUNvbnRleHQoZ2wpO1xuICAgICAgfVxuICAgICAgdGhpcy5iaW5hcnlDYWNoZSA9IHt9O1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gZmFsc2U7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpKTtcbiAgICAgIG5ld0dQR1BVID0gbmV3IEdQR1BVQ29udGV4dChnbCk7XG4gICAgICB0aGlzLmJpbmFyeUNhY2hlID0gZ2V0QmluYXJ5Q2FjaGUoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9WRVJTSU9OJykpO1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gdHJ1ZTtcbiAgICB9XG5cbiAgICB0aGlzLmdwZ3B1ID0gbmV3R1BHUFU7XG4gICAgdGhpcy5jYW52YXMgPSB0aGlzLmdwZ3B1LmdsLmNhbnZhcztcbiAgICB0aGlzLnRleHR1cmVNYW5hZ2VyID0gbmV3IFRleHR1cmVNYW5hZ2VyKHRoaXMuZ3BncHUpO1xuICAgIHRoaXMubnVtTUJCZWZvcmVXYXJuaW5nID0gbnVtTUJCZWZvcmVXYXJuaW5nKCk7XG4gICAgdGhpcy50ZXhEYXRhID0gbmV3IERhdGFTdG9yYWdlKHRoaXMsIGVuZ2luZSgpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIG51bURhdGFJZHMoKSB7XG4gICAgcmV0dXJuIHRoaXMudGV4RGF0YS5udW1EYXRhSWRzKCkgLSB0aGlzLnBlbmRpbmdEZWxldGVzO1xuICB9XG5cbiAgLy8gV3JpdGVzIGEgbmV3IGVudHJ5IHRvIHRoZSBkYXRhIHN0b3JlIHdpdGggYSBXZWJHTCB0ZXh0dXJlLCBhbmQgcmVnaXN0ZXJzIGl0XG4gIC8vIHRvIHRoZSB0ZXh0dXJlIG1hbmFnZXIuXG4gIHdyaXRlVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICB0ZXhIZWlnaHQ6IG51bWJlciwgdGV4V2lkdGg6IG51bWJlciwgY2hhbm5lbHM6IHN0cmluZyk6IERhdGFJZCB7XG4gICAgLy8gVGVtcG9yYXJpbHkgY3JlYXRlIGFuIHRlbnNvciBpbmZvIHRvIG1ha2UgdGhlIHRleHR1cmUgY29tcGF0aWJsZSB3aXRoXG4gICAgLy8gdGhlIHJ1bldlYkdMUHJvZ3JhbSdzIGlucHV0LlxuICAgIGNvbnN0IGlucHV0ID0gdGhpcy5tYWtlVGVuc29ySW5mbyhzaGFwZSwgZHR5cGUpO1xuICAgIGNvbnN0IGluRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoaW5wdXQuZGF0YUlkKTtcbiAgICAvLyBFdmVuIHRob3VnaCB0aGUgaW5wdXQgdGV4dHVyZSBjb3VsZCBiZSB1bnBhY2tlZCBvciBkZW5zZSBwYWNrZWQsIGl0IGlzXG4gICAgLy8gYWx3YXlzIGNvbnNpZGVyZWQgYXMgdW5wYWNrZWQgZm9yIEVuY29kZU1hdHJpeFByb2dyYW0uXG4gICAgaW5EYXRhLmlzUGFja2VkID0gZmFsc2U7XG5cbiAgICAvLyBCaW5kIHRleHR1cmUgdG8gdGhlIGlucHV0IHRlbnNvci5cbiAgICBpbkRhdGEudGV4dHVyZSA9IHt0ZXh0dXJlLCB0ZXhTaGFwZTogW3RleEhlaWdodCwgdGV4V2lkdGhdfTtcbiAgICBpbkRhdGEudGV4U2hhcGUgPSBbdGV4SGVpZ2h0LCB0ZXhXaWR0aF07XG5cbiAgICBjb25zdCBzaGFwZUFzM0QgPSB3ZWJnbF91dGlsLmdldFNoYXBlQXMzRChzaGFwZSk7XG4gICAgY29uc3QgcHJvZ3JhbSA9XG4gICAgICAgIG5ldyBFbmNvZGVNYXRyaXhQcm9ncmFtKHNoYXBlQXMzRCwgZmFsc2UgLyogaXNCeXRlQXJyYXkgKi8sIGNoYW5uZWxzKTtcbiAgICBjb25zdCBvdXRwdXQgPVxuICAgICAgICB0aGlzLnJ1bldlYkdMUHJvZ3JhbShwcm9ncmFtLCBbaW5wdXRdLCBkdHlwZSwgW1t0ZXhIZWlnaHQsIHRleFdpZHRoXV0pO1xuICAgIG91dHB1dC5zaGFwZSA9IHNoYXBlO1xuXG4gICAgLy8gVW5iaW5kIHRoZSB0ZXh0dXJlIGZyb20gdGhlIGlucHV0IHRlbnNvciB0byBhdm9pZCB0aGUgdGV4dHVyZSBiZWluZ1xuICAgIC8vIHJlbGVhc2VkLlxuICAgIGluRGF0YS50ZXh0dXJlID0gbnVsbDtcbiAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKGlucHV0KTtcblxuICAgIHJldHVybiBvdXRwdXQuZGF0YUlkO1xuICB9XG5cbiAgb3ZlcnJpZGUgd3JpdGUodmFsdWVzOiBCYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6XG4gICAgICBEYXRhSWQge1xuICAgIGlmIChlbnYoKS5nZXRCb29sKCdXRUJHTF9DSEVDS19OVU1FUklDQUxfUFJPQkxFTVMnKSB8fFxuICAgICAgICBlbnYoKS5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgICB0aGlzLmNoZWNrTnVtZXJpY2FsUHJvYmxlbXModmFsdWVzKTtcbiAgICB9XG4gICAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0JyAmJiB2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBDYW5ub3Qgd3JpdGUgdG8gYSBjb21wbGV4NjQgZHR5cGUuIGAgK1xuICAgICAgICAgIGBQbGVhc2UgdXNlIHRmLmNvbXBsZXgocmVhbCwgaW1hZykuYCk7XG4gICAgfVxuICAgIGNvbnN0IGRhdGFJZCA9IHtpZDogdGhpcy5uZXh0RGF0YUlkKCl9O1xuICAgIHRoaXMudGV4RGF0YS5zZXQoXG4gICAgICAgIGRhdGFJZCxcbiAgICAgICAge3NoYXBlLCBkdHlwZSwgdmFsdWVzLCB1c2FnZTogVGV4dHVyZVVzYWdlLlVQTE9BRCwgcmVmQ291bnQ6IDF9KTtcbiAgICByZXR1cm4gZGF0YUlkO1xuICB9XG5cbiAgLyoqIFJldHVybiByZWZDb3VudCBvZiBhIGBUZW5zb3JEYXRhYC4gKi9cbiAgb3ZlcnJpZGUgcmVmQ291bnQoZGF0YUlkOiBEYXRhSWQpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnRleERhdGEuaGFzKGRhdGFJZCkpIHtcbiAgICAgIGNvbnN0IHRlbnNvckRhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgICByZXR1cm4gdGVuc29yRGF0YS5yZWZDb3VudDtcbiAgICB9XG4gICAgcmV0dXJuIDA7XG4gIH1cblxuICAvKiogSW5jcmVhc2UgcmVmQ291bnQgb2YgYSBgVGV4dHVyZURhdGFgLiAqL1xuICBvdmVycmlkZSBpbmNSZWYoZGF0YUlkOiBEYXRhSWQpOiB2b2lkIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIHRleERhdGEucmVmQ291bnQrKztcbiAgfVxuXG4gIC8qKiBEZWNyZWFzZSByZWZDb3VudCBvZiBhIGBUZXh0dXJlRGF0YWAuICovXG4gIGRlY1JlZihkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGlmICh0aGlzLnRleERhdGEuaGFzKGRhdGFJZCkpIHtcbiAgICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgICB0ZXhEYXRhLnJlZkNvdW50LS07XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgbW92ZShcbiAgICAgIGRhdGFJZDogRGF0YUlkLCB2YWx1ZXM6IEJhY2tlbmRWYWx1ZXMsIHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLFxuICAgICAgcmVmQ291bnQ6IG51bWJlcik6IHZvaWQge1xuICAgIGlmIChlbnYoKS5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgICB0aGlzLmNoZWNrTnVtZXJpY2FsUHJvYmxlbXModmFsdWVzKTtcbiAgICB9XG4gICAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBDYW5ub3Qgd3JpdGUgdG8gYSBjb21wbGV4NjQgZHR5cGUuIGAgK1xuICAgICAgICAgIGBQbGVhc2UgdXNlIHRmLmNvbXBsZXgocmVhbCwgaW1hZykuYCk7XG4gICAgfVxuICAgIHRoaXMudGV4RGF0YS5zZXQoXG4gICAgICAgIGRhdGFJZCwge3NoYXBlLCBkdHlwZSwgdmFsdWVzLCB1c2FnZTogVGV4dHVyZVVzYWdlLlVQTE9BRCwgcmVmQ291bnR9KTtcbiAgfVxuXG4gIGRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKHRlbnNvckluZm86IFRlbnNvckluZm8pOiB2b2lkIHtcbiAgICB0aGlzLmRpc3Bvc2VEYXRhKHRlbnNvckluZm8uZGF0YUlkKTtcbiAgfVxuXG4gIG92ZXJyaWRlIHJlYWRTeW5jKGRhdGFJZDogRGF0YUlkKTogQmFja2VuZFZhbHVlcyB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7dmFsdWVzLCBkdHlwZSwgY29tcGxleFRlbnNvckluZm9zLCBzbGljZSwgc2hhcGUsIGlzUGFja2VkfSA9IHRleERhdGE7XG5cbiAgICAvLyBUaGUgcHJlc2VuY2Ugb2YgYHNsaWNlYCBpbmRpY2F0ZXMgdGhpcyB0ZW5zb3IgaXMgYSBzaGFsbG93IHNsaWNlIG9mIGFcbiAgICAvLyBkaWZmZXJlbnQgdGVuc29yLCBhbmQgaXMgdXNpbmcgdGhhdCBvcmlnaW5hbCB0ZW5zb3IncyB0ZXh0dXJlLiBSdW5cbiAgICAvLyBgY2xvbmVgIGluIG9yZGVyIHRvIGNvcHkgdGhhdCB0ZXh0dXJlIGFuZCByZWFkIGZyb20gaXQuXG4gICAgaWYgKHNsaWNlICE9IG51bGwpIHtcbiAgICAgIGxldCBwcm9ncmFtO1xuICAgICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFBhY2tlZFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHJlcyA9XG4gICAgICAgICAgdGhpcy5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW3tkYXRhSWQsIHNoYXBlLCBkdHlwZX1dLCBkdHlwZSk7XG4gICAgICBjb25zdCBkYXRhID0gdGhpcy5yZWFkU3luYyhyZXMuZGF0YUlkKTtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ocmVzKTtcbiAgICAgIHJldHVybiBkYXRhO1xuICAgIH1cbiAgICBpZiAodmFsdWVzICE9IG51bGwpIHtcbiAgICAgIHJldHVybiB0aGlzLmNvbnZlcnRBbmRDYWNoZU9uQ1BVKGRhdGFJZCk7XG4gICAgfVxuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHJldHVybiB2YWx1ZXM7XG4gICAgfVxuICAgIGNvbnN0IHNob3VsZFRpbWVQcm9ncmFtID0gdGhpcy5hY3RpdmVUaW1lcnMgIT0gbnVsbDtcbiAgICBsZXQgc3RhcnQ6IG51bWJlcjtcbiAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgIHN0YXJ0ID0gdXRpbC5ub3coKTtcbiAgICB9XG5cbiAgICBsZXQgcmVzdWx0OiBGbG9hdDMyQXJyYXk7XG4gICAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgY29uc3QgcmVhbFZhbHVlcyA9XG4gICAgICAgICAgdGhpcy5yZWFkU3luYyhjb21wbGV4VGVuc29ySW5mb3MucmVhbC5kYXRhSWQpIGFzIEZsb2F0MzJBcnJheTtcbiAgICAgIGNvbnN0IGltYWdWYWx1ZXMgPVxuICAgICAgICAgIHRoaXMucmVhZFN5bmMoY29tcGxleFRlbnNvckluZm9zLmltYWcuZGF0YUlkKSBhcyBGbG9hdDMyQXJyYXk7XG4gICAgICByZXN1bHQgPSBiYWNrZW5kX3V0aWwubWVyZ2VSZWFsQW5kSW1hZ0FycmF5cyhyZWFsVmFsdWVzLCBpbWFnVmFsdWVzKTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmVzdWx0ID0gdGhpcy5nZXRWYWx1ZXNGcm9tVGV4dHVyZShkYXRhSWQpO1xuICAgIH1cblxuICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgdGhpcy5kb3dubG9hZFdhaXRNcyArPSB1dGlsLm5vdygpIC0gc3RhcnQ7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmNvbnZlcnRBbmRDYWNoZU9uQ1BVKGRhdGFJZCwgcmVzdWx0KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIHJlYWQoZGF0YUlkOiBEYXRhSWQpOiBQcm9taXNlPEJhY2tlbmRWYWx1ZXM+IHtcbiAgICBpZiAodGhpcy5wZW5kaW5nUmVhZC5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3Qgc3Vic2NyaWJlcnMgPSB0aGlzLnBlbmRpbmdSZWFkLmdldChkYXRhSWQpO1xuICAgICAgcmV0dXJuIG5ldyBQcm9taXNlPFR5cGVkQXJyYXk+KHJlc29sdmUgPT4gc3Vic2NyaWJlcnMucHVzaChyZXNvbHZlKSk7XG4gICAgfVxuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qge3ZhbHVlcywgc2hhcGUsIHNsaWNlLCBkdHlwZSwgY29tcGxleFRlbnNvckluZm9zLCBpc1BhY2tlZH0gPSB0ZXhEYXRhO1xuXG4gICAgLy8gVGhlIHByZXNlbmNlIG9mIGBzbGljZWAgaW5kaWNhdGVzIHRoaXMgdGVuc29yIGlzIGEgc2hhbGxvdyBzbGljZSBvZiBhXG4gICAgLy8gZGlmZmVyZW50IHRlbnNvciwgYW5kIGlzIHVzaW5nIHRoYXQgb3JpZ2luYWwgdGVuc29yJ3MgdGV4dHVyZS4gUnVuXG4gICAgLy8gYGNsb25lYCBpbiBvcmRlciB0byBjb3B5IHRoYXQgdGV4dHVyZSBhbmQgcmVhZCBmcm9tIGl0LlxuICAgIGlmIChzbGljZSAhPSBudWxsKSB7XG4gICAgICBsZXQgcHJvZ3JhbTtcbiAgICAgIGlmIChpc1BhY2tlZCkge1xuICAgICAgICBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQYWNrZWRQcm9ncmFtKHNoYXBlLCB1bmFyeV9vcC5DTE9ORSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQcm9ncmFtKHNoYXBlLCB1bmFyeV9vcC5DTE9ORSk7XG4gICAgICB9XG4gICAgICBjb25zdCByZXMgPVxuICAgICAgICAgIHRoaXMucnVuV2ViR0xQcm9ncmFtKHByb2dyYW0sIFt7ZGF0YUlkLCBzaGFwZSwgZHR5cGV9XSwgZHR5cGUpO1xuICAgICAgY29uc3QgZGF0YSA9IHRoaXMucmVhZChyZXMuZGF0YUlkKTtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ocmVzKTtcbiAgICAgIHJldHVybiBkYXRhO1xuICAgIH1cblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkKTtcbiAgICB9XG5cbiAgICBpZiAoZW52KCkuZ2V0Qm9vbCgnREVCVUcnKSkge1xuICAgICAgLy8gZ2V0Qm9vbCgnV0VCR0xfRE9XTkxPQURfRkxPQVRfRU5BQkxFRCcpIGNhdXNlZCBhIGJsb2NraW5nIEdQVSBjYWxsLlxuICAgICAgLy8gRm9yIHBlcmZvcm1hbmNlIHJlYXNvbiwgb25seSBjaGVjayBpdCBmb3IgZGVidWdnaW5nLiBJbiBwcm9kdWN0aW9uLFxuICAgICAgLy8gaXQgZG9lc24ndCBoYW5kbGUgdGhpcyB1c2UgY2FzZSBhbnl3YXksIHNvIGJlaGF2aW9yIGlzIG5vdCBjaGFuZ2VkLlxuICAgICAgaWYgKCFlbnYoKS5nZXRCb29sKCdXRUJHTF9ET1dOTE9BRF9GTE9BVF9FTkFCTEVEJykgJiZcbiAgICAgICAgICBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSA9PT0gMikge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgdGVuc29yLmRhdGEoKSB3aXRoIFdFQkdMX0RPV05MT0FEX0ZMT0FUX0VOQUJMRUQ9ZmFsc2UgYW5kIGAgK1xuICAgICAgICAgICAgYFdFQkdMX1ZFUlNJT049MiBub3QgeWV0IHN1cHBvcnRlZC5gKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBsZXQgYnVmZmVyOiBXZWJHTEJ1ZmZlciA9IG51bGw7XG4gICAgbGV0IHRtcERvd25sb2FkVGFyZ2V0OiBUZW5zb3JJbmZvO1xuXG4gICAgaWYgKGR0eXBlICE9PSAnY29tcGxleDY0JyAmJiBlbnYoKS5nZXQoJ1dFQkdMX0JVRkZFUl9TVVBQT1JURUQnKSkge1xuICAgICAgLy8gUG9zc2libHkgY29weSB0aGUgdGV4dHVyZSBpbnRvIGEgYnVmZmVyIGJlZm9yZSBpbnNlcnRpbmcgYSBmZW5jZS5cbiAgICAgIHRtcERvd25sb2FkVGFyZ2V0ID0gdGhpcy5kZWNvZGUoZGF0YUlkKTtcbiAgICAgIGNvbnN0IHRtcERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KHRtcERvd25sb2FkVGFyZ2V0LmRhdGFJZCk7XG5cbiAgICAgIGJ1ZmZlciA9IHRoaXMuZ3BncHUuY3JlYXRlQnVmZmVyRnJvbVRleHR1cmUoXG4gICAgICAgICAgdG1wRGF0YS50ZXh0dXJlLnRleHR1cmUsIC4uLnRleF91dGlsLmdldERlbnNlVGV4U2hhcGUoc2hhcGUpKTtcbiAgICB9XG5cbiAgICB0aGlzLnBlbmRpbmdSZWFkLnNldChkYXRhSWQsIFtdKTtcblxuICAgIGlmIChkdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIC8vIENyZWF0ZSBhIGZlbmNlIGFuZCB3YWl0IGZvciBpdCB0byByZXNvbHZlLlxuICAgICAgYXdhaXQgdGhpcy5ncGdwdS5jcmVhdGVBbmRXYWl0Rm9yRmVuY2UoKTtcbiAgICB9XG5cbiAgICAvLyBEb3dubG9hZCB0aGUgdmFsdWVzIGZyb20gdGhlIEdQVS5cbiAgICBsZXQgdmFsczogRmxvYXQzMkFycmF5O1xuICAgIGlmIChkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIGNvbnN0IHBzID0gYXdhaXQgUHJvbWlzZS5hbGwoW1xuICAgICAgICB0aGlzLnJlYWQoY29tcGxleFRlbnNvckluZm9zLnJlYWwuZGF0YUlkKSxcbiAgICAgICAgdGhpcy5yZWFkKGNvbXBsZXhUZW5zb3JJbmZvcy5pbWFnLmRhdGFJZClcbiAgICAgIF0pO1xuXG4gICAgICBjb25zdCByZWFsVmFsdWVzID0gcHNbMF07XG4gICAgICBjb25zdCBpbWFnVmFsdWVzID0gcHNbMV07XG4gICAgICB2YWxzID0gYmFja2VuZF91dGlsLm1lcmdlUmVhbEFuZEltYWdBcnJheXMoXG4gICAgICAgICAgcmVhbFZhbHVlcyBhcyBGbG9hdDMyQXJyYXksIGltYWdWYWx1ZXMgYXMgRmxvYXQzMkFycmF5KTtcbiAgICB9IGVsc2UgaWYgKGJ1ZmZlciA9PSBudWxsKSB7XG4gICAgICB2YWxzID0gdGhpcy5nZXRWYWx1ZXNGcm9tVGV4dHVyZShkYXRhSWQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICAgIHZhbHMgPSB0aGlzLmdwZ3B1LmRvd25sb2FkRmxvYXQzMk1hdHJpeEZyb21CdWZmZXIoYnVmZmVyLCBzaXplKTtcbiAgICB9XG4gICAgaWYgKHRtcERvd25sb2FkVGFyZ2V0ICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odG1wRG93bmxvYWRUYXJnZXQpO1xuICAgIH1cbiAgICBpZiAoYnVmZmVyICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IGdsID0gdGhpcy5ncGdwdS5nbDtcbiAgICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVCdWZmZXIoYnVmZmVyKSk7XG4gICAgfVxuICAgIGNvbnN0IGRUeXBlVmFscyA9IHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkLCB2YWxzKTtcblxuICAgIGNvbnN0IHN1YnNjcmliZXJzID0gdGhpcy5wZW5kaW5nUmVhZC5nZXQoZGF0YUlkKTtcbiAgICB0aGlzLnBlbmRpbmdSZWFkLmRlbGV0ZShkYXRhSWQpO1xuXG4gICAgLy8gTm90aWZ5IGFsbCBwZW5kaW5nIHJlYWRzLlxuICAgIHN1YnNjcmliZXJzLmZvckVhY2gocmVzb2x2ZSA9PiByZXNvbHZlKGRUeXBlVmFscykpO1xuICAgIGlmICh0aGlzLnBlbmRpbmdEaXNwb3NhbC5oYXMoZGF0YUlkKSkge1xuICAgICAgdGhpcy5wZW5kaW5nRGlzcG9zYWwuZGVsZXRlKGRhdGFJZCk7XG4gICAgICBpZiAodGhpcy5kaXNwb3NlRGF0YShkYXRhSWQpKSB7XG4gICAgICAgIGVuZ2luZSgpLnJlbW92ZURhdGFJZChkYXRhSWQsIHRoaXMpO1xuICAgICAgfVxuICAgICAgdGhpcy5wZW5kaW5nRGVsZXRlcy0tO1xuICAgIH1cbiAgICByZXR1cm4gZFR5cGVWYWxzO1xuICB9XG5cbiAgLyoqXG4gICAqIFJlYWQgdGVuc29yIHRvIGEgbmV3IHRleHR1cmUgdGhhdCBpcyBkZW5zZWx5IHBhY2tlZCBmb3IgZWFzZSBvZiB1c2UuXG4gICAqIEBwYXJhbSBkYXRhSWQgVGhlIHNvdXJjZSB0ZW5zb3IuXG4gICAqIEBwYXJhbSBvcHRpb25zXG4gICAqICAgICBjdXN0b21UZXhTaGFwZTogT3B0aW9uYWwuIElmIHNldCwgd2lsbCB1c2UgdGhlIHVzZXIgZGVmaW5lZCB0ZXh0dXJlXG4gICAqICAgICBzaGFwZSB0byBjcmVhdGUgdGhlIHRleHR1cmUuXG4gICAqL1xuICBvdmVycmlkZSByZWFkVG9HUFUoZGF0YUlkOiBEYXRhSWQsIG9wdGlvbnM6IERhdGFUb0dQVVdlYkdMT3B0aW9uID0ge30pOlxuICAgICAgR1BVRGF0YSB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7dmFsdWVzLCBzaGFwZSwgc2xpY2UsIGR0eXBlLCBpc1BhY2tlZCwgdGV4dHVyZX0gPSB0ZXhEYXRhO1xuXG4gICAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdEb2VzIG5vdCBzdXBwb3J0IHJlYWRpbmcgdGV4dHVyZSBmb3IgY29tcGxleDY0IGR0eXBlLicpO1xuICAgIH1cblxuICAgIC8vIFRoZSBwcmVzZW5jZSBvZiBgc2xpY2VgIGluZGljYXRlcyB0aGlzIHRlbnNvciBpcyBhIHNoYWxsb3cgc2xpY2Ugb2YgYVxuICAgIC8vIGRpZmZlcmVudCB0ZW5zb3IsIGFuZCBpcyB1c2luZyB0aGF0IG9yaWdpbmFsIHRlbnNvcidzIHRleHR1cmUuIFJ1blxuICAgIC8vIGBjbG9uZWAgaW4gb3JkZXIgdG8gY29weSB0aGF0IHRleHR1cmUgYW5kIHJlYWQgZnJvbSBpdC5cbiAgICBpZiAoc2xpY2UgIT0gbnVsbCkge1xuICAgICAgbGV0IHByb2dyYW07XG4gICAgICBpZiAoaXNQYWNrZWQpIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUGFja2VkUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfVxuICAgICAgY29uc3QgcmVzID1cbiAgICAgICAgICB0aGlzLnJ1bldlYkdMUHJvZ3JhbShwcm9ncmFtLCBbe2RhdGFJZCwgc2hhcGUsIGR0eXBlfV0sIGR0eXBlKTtcbiAgICAgIGNvbnN0IGdwdVJlc291b3JjZSA9IHRoaXMucmVhZFRvR1BVKHJlcywgb3B0aW9ucyk7XG4gICAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKHJlcyk7XG4gICAgICByZXR1cm4gZ3B1UmVzb3VvcmNlO1xuICAgIH1cblxuICAgIGlmICh0ZXh0dXJlID09IG51bGwpIHtcbiAgICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0RhdGEgaXMgbm90IG9uIEdQVSBidXQgb24gQ1BVLicpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdUaGVyZSBpcyBubyBkYXRhIG9uIEdQVSBvciBDUFUuJyk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gRGVjb2RlIHRoZSB0ZXh0dXJlIHNvIHRoYXQgaXQgaXMgc3RvcmVkIGRlbnNlbHkgKHVzaW5nIGZvdXIgY2hhbm5lbHMpLlxuICAgIGNvbnN0IHRtcFRhcmdldCA9IHRoaXMuZGVjb2RlKGRhdGFJZCwgb3B0aW9ucy5jdXN0b21UZXhTaGFwZSk7XG5cbiAgICAvLyBNYWtlIGVuZ2luZSB0cmFjayB0aGlzIHRlbnNvciwgc28gdGhhdCB3ZSBjYW4gZGlzcG9zZSBpdCBsYXRlci5cbiAgICBjb25zdCB0ZW5zb3JSZWYgPSBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbVRlbnNvckluZm8odG1wVGFyZ2V0KTtcblxuICAgIGNvbnN0IHRtcERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KHRtcFRhcmdldC5kYXRhSWQpO1xuICAgIHJldHVybiB7dGVuc29yUmVmLCAuLi50bXBEYXRhLnRleHR1cmV9O1xuICB9XG5cbiAgYnVmZmVyU3luYzxSIGV4dGVuZHMgUmFuaywgRCBleHRlbmRzIERhdGFUeXBlPih0OiBUZW5zb3JJbmZvKTpcbiAgICAgIFRlbnNvckJ1ZmZlcjxSLCBEPiB7XG4gICAgY29uc3QgZGF0YSA9IHRoaXMucmVhZFN5bmModC5kYXRhSWQpO1xuICAgIGlmICh0LmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgdHJ5IHtcbiAgICAgICAgLy8gRGVjb2RlIHRoZSBieXRlcyBpbnRvIHN0cmluZy5cbiAgICAgICAgY29uc3Qgc3RyaW5ncyA9IChkYXRhIGFzIFVpbnQ4QXJyYXlbXSkubWFwKGQgPT4gdXRpbC5kZWNvZGVTdHJpbmcoZCkpO1xuICAgICAgICByZXR1cm4gYnVmZmVyKHQuc2hhcGUgYXMgU2hhcGVNYXBbUl0sIHQuZHR5cGUsIHN0cmluZ3MpIGFzXG4gICAgICAgICAgICBUZW5zb3JCdWZmZXI8UiwgRD47XG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gZGVjb2RlIGVuY29kZWQgc3RyaW5nIGJ5dGVzIGludG8gdXRmLTgnKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGJ1ZmZlcih0LnNoYXBlIGFzIFNoYXBlTWFwW1JdLCB0LmR0eXBlLCBkYXRhIGFzIFR5cGVkQXJyYXkpIGFzXG4gICAgICAgIFRlbnNvckJ1ZmZlcjxSLCBEPjtcbiAgfVxuXG4gIHByaXZhdGUgY2hlY2tOdW1lcmljYWxQcm9ibGVtcyh2YWx1ZXM6IEJhY2tlbmRWYWx1ZXMpOiB2b2lkIHtcbiAgICBpZiAodmFsdWVzID09IG51bGwpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IG51bSA9IHZhbHVlc1tpXSBhcyBudW1iZXI7XG4gICAgICBpZiAoIXdlYmdsX3V0aWwuY2FuQmVSZXByZXNlbnRlZChudW0pKSB7XG4gICAgICAgIGlmIChlbnYoKS5nZXRCb29sKCdXRUJHTF9SRU5ERVJfRkxPQVQzMl9DQVBBQkxFJykpIHtcbiAgICAgICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICAgICAgYFRoZSB2YWx1ZSAke251bX0gY2Fubm90IGJlIHJlcHJlc2VudGVkIHdpdGggeW91ciBgICtcbiAgICAgICAgICAgICAgYGN1cnJlbnQgc2V0dGluZ3MuIENvbnNpZGVyIGVuYWJsaW5nIGZsb2F0MzIgcmVuZGVyaW5nOiBgICtcbiAgICAgICAgICAgICAgYCd0Zi5lbnYoKS5zZXQoJ1dFQkdMX1JFTkRFUl9GTE9BVDMyX0VOQUJMRUQnLCB0cnVlKTsnYCk7XG4gICAgICAgIH1cbiAgICAgICAgdGhyb3cgRXJyb3IoYFRoZSB2YWx1ZSAke251bX0gY2Fubm90IGJlIHJlcHJlc2VudGVkIG9uIHRoaXMgZGV2aWNlLmApO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZ2V0VmFsdWVzRnJvbVRleHR1cmUoZGF0YUlkOiBEYXRhSWQpOiBGbG9hdDMyQXJyYXkge1xuICAgIGNvbnN0IHtzaGFwZSwgZHR5cGUsIGlzUGFja2VkfSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICBpZiAoZW52KCkuZ2V0Qm9vbCgnV0VCR0xfRE9XTkxPQURfRkxPQVRfRU5BQkxFRCcpKSB7XG4gICAgICBjb25zdCB0bXBUYXJnZXQgPSB0aGlzLmRlY29kZShkYXRhSWQpO1xuICAgICAgY29uc3QgdG1wRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQodG1wVGFyZ2V0LmRhdGFJZCk7XG4gICAgICBjb25zdCB2YWxzID1cbiAgICAgICAgICB0aGlzLmdwZ3B1XG4gICAgICAgICAgICAgIC5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgICAgICAgICAgICAgdG1wRGF0YS50ZXh0dXJlLnRleHR1cmUsIC4uLnRleF91dGlsLmdldERlbnNlVGV4U2hhcGUoc2hhcGUpKVxuICAgICAgICAgICAgICAuc3ViYXJyYXkoMCwgc2l6ZSk7XG5cbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odG1wVGFyZ2V0KTtcblxuICAgICAgcmV0dXJuIHZhbHM7XG4gICAgfVxuXG4gICAgY29uc3Qgc2hvdWxkVXNlUGFja2VkUHJvZ3JhbSA9XG4gICAgICAgIGVudigpLmdldEJvb2woJ1dFQkdMX1BBQ0snKSAmJiBpc1BhY2tlZCA9PT0gdHJ1ZTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9XG4gICAgICAgIHNob3VsZFVzZVBhY2tlZFByb2dyYW0gPyB3ZWJnbF91dGlsLmdldFNoYXBlQXMzRChzaGFwZSkgOiBzaGFwZTtcbiAgICBjb25zdCBwcm9ncmFtID0gc2hvdWxkVXNlUGFja2VkUHJvZ3JhbSA/XG4gICAgICAgIG5ldyBFbmNvZGVGbG9hdFBhY2tlZFByb2dyYW0ob3V0cHV0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdKSA6XG4gICAgICAgIG5ldyBFbmNvZGVGbG9hdFByb2dyYW0ob3V0cHV0U2hhcGUpO1xuICAgIGNvbnN0IG91dHB1dCA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBbe3NoYXBlOiBvdXRwdXRTaGFwZSwgZHR5cGUsIGRhdGFJZH1dLCAnZmxvYXQzMicpO1xuICAgIGNvbnN0IHRtcERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KG91dHB1dC5kYXRhSWQpO1xuICAgIGNvbnN0IHZhbHMgPSB0aGlzLmdwZ3B1XG4gICAgICAgICAgICAgICAgICAgICAuZG93bmxvYWRCeXRlRW5jb2RlZEZsb2F0TWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgICAgICAgICAgICAgICAgICAgICAgdG1wRGF0YS50ZXh0dXJlLnRleHR1cmUsIHRtcERhdGEudGV4U2hhcGVbMF0sXG4gICAgICAgICAgICAgICAgICAgICAgICAgdG1wRGF0YS50ZXhTaGFwZVsxXSlcbiAgICAgICAgICAgICAgICAgICAgIC5zdWJhcnJheSgwLCBzaXplKTtcbiAgICB0aGlzLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKG91dHB1dCk7XG5cbiAgICByZXR1cm4gdmFscztcbiAgfVxuXG4gIG92ZXJyaWRlIHRpbWVyQXZhaWxhYmxlKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMDtcbiAgfVxuXG4gIG92ZXJyaWRlIHRpbWUoZjogKCkgPT4gdm9pZCk6IFByb21pc2U8V2ViR0xUaW1pbmdJbmZvPiB7XG4gICAgY29uc3Qgb2xkQWN0aXZlVGltZXJzID0gdGhpcy5hY3RpdmVUaW1lcnM7XG4gICAgY29uc3QgbmV3QWN0aXZlVGltZXJzOiBUaW1lck5vZGVbXSA9IFtdO1xuXG4gICAgbGV0IG91dGVyTW9zdFRpbWUgPSBmYWxzZTtcbiAgICBpZiAodGhpcy5wcm9ncmFtVGltZXJzU3RhY2sgPT0gbnVsbCkge1xuICAgICAgdGhpcy5wcm9ncmFtVGltZXJzU3RhY2sgPSBuZXdBY3RpdmVUaW1lcnM7XG4gICAgICBvdXRlck1vc3RUaW1lID0gdHJ1ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5hY3RpdmVUaW1lcnMucHVzaChuZXdBY3RpdmVUaW1lcnMpO1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVRpbWVycyA9IG5ld0FjdGl2ZVRpbWVycztcblxuICAgIGYoKTtcblxuICAgIC8vIG5lZWRpbmcgdG8gc3BsaXQgdGhlc2UgdXAgYmVjYXVzZSB1dGlsLmZsYXR0ZW4gb25seSBhY2NlcHRzIGNlcnRhaW4gdHlwZXNcbiAgICBjb25zdCBmbGF0dGVuZWRBY3RpdmVUaW1lclF1ZXJpZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBLZXJuZWxJbmZvKSA9PiBkLnF1ZXJ5KSlcbiAgICAgICAgICAgIC5maWx0ZXIoZCA9PiBkICE9IG51bGwpO1xuICAgIGNvbnN0IGZsYXR0ZW5lZEFjdGl2ZVRpbWVyTmFtZXMgPVxuICAgICAgICB1dGlsLmZsYXR0ZW4odGhpcy5hY3RpdmVUaW1lcnMubWFwKChkOiBLZXJuZWxJbmZvKSA9PiBkLm5hbWUpKVxuICAgICAgICAgICAgLmZpbHRlcihkID0+IGQgIT0gbnVsbCk7XG5cbiAgICB0aGlzLmFjdGl2ZVRpbWVycyA9IG9sZEFjdGl2ZVRpbWVycztcblxuICAgIGlmIChvdXRlck1vc3RUaW1lKSB7XG4gICAgICB0aGlzLnByb2dyYW1UaW1lcnNTdGFjayA9IG51bGw7XG4gICAgfVxuXG4gICAgY29uc3QgcmVzOiBXZWJHTFRpbWluZ0luZm8gPSB7XG4gICAgICB1cGxvYWRXYWl0TXM6IHRoaXMudXBsb2FkV2FpdE1zLFxuICAgICAgZG93bmxvYWRXYWl0TXM6IHRoaXMuZG93bmxvYWRXYWl0TXMsXG4gICAgICBrZXJuZWxNczogbnVsbCxcbiAgICAgIHdhbGxNczogbnVsbCAgLy8gd2lsbCBiZSBmaWxsZWQgYnkgdGhlIGVuZ2luZVxuICAgIH07XG5cbiAgICByZXR1cm4gKGFzeW5jICgpID0+IHtcbiAgICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID5cbiAgICAgICAgICAwKSB7XG4gICAgICAgIGNvbnN0IGtlcm5lbE1zID0gYXdhaXQgUHJvbWlzZS5hbGwoZmxhdHRlbmVkQWN0aXZlVGltZXJRdWVyaWVzKTtcblxuICAgICAgICByZXNbJ2tlcm5lbE1zJ10gPSB1dGlsLnN1bShrZXJuZWxNcyk7XG4gICAgICAgIHJlc1snZ2V0RXh0cmFQcm9maWxlSW5mbyddID0gKCkgPT5cbiAgICAgICAgICAgIGtlcm5lbE1zXG4gICAgICAgICAgICAgICAgLm1hcCgoZCwgaSkgPT4gKHtuYW1lOiBmbGF0dGVuZWRBY3RpdmVUaW1lck5hbWVzW2ldLCBtczogZH0pKVxuICAgICAgICAgICAgICAgIC5tYXAoZCA9PiBgJHtkLm5hbWV9OiAke2QubXN9YClcbiAgICAgICAgICAgICAgICAuam9pbignLCAnKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJlc1sna2VybmVsTXMnXSA9IHtcbiAgICAgICAgICBlcnJvcjogJ1dlYkdMIHF1ZXJ5IHRpbWVycyBhcmUgbm90IHN1cHBvcnRlZCBpbiB0aGlzIGVudmlyb25tZW50LidcbiAgICAgICAgfTtcbiAgICAgIH1cblxuICAgICAgdGhpcy51cGxvYWRXYWl0TXMgPSAwO1xuICAgICAgdGhpcy5kb3dubG9hZFdhaXRNcyA9IDA7XG4gICAgICByZXR1cm4gcmVzO1xuICAgIH0pKCk7XG4gIH1cbiAgb3ZlcnJpZGUgbWVtb3J5KCk6IFdlYkdMTWVtb3J5SW5mbyB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHVucmVsaWFibGU6IGZhbHNlLFxuICAgICAgbnVtQnl0ZXNJbkdQVTogdGhpcy5udW1CeXRlc0luR1BVLFxuICAgICAgbnVtQnl0ZXNJbkdQVUFsbG9jYXRlZDogdGhpcy50ZXh0dXJlTWFuYWdlci5udW1CeXRlc0FsbG9jYXRlZCxcbiAgICAgIG51bUJ5dGVzSW5HUFVGcmVlOiB0aGlzLnRleHR1cmVNYW5hZ2VyLm51bUJ5dGVzRnJlZVxuICAgIH0gYXMgV2ViR0xNZW1vcnlJbmZvO1xuICB9XG5cbiAgcHJpdmF0ZSBzdGFydFRpbWVyKCk6IFdlYkdMUXVlcnl8Q1BVVGltZXJRdWVyeSB7XG4gICAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1JFTElBQkxFJykgPiAwKSB7XG4gICAgICByZXR1cm4gdGhpcy5ncGdwdS5iZWdpblF1ZXJ5KCk7XG4gICAgfVxuICAgIHJldHVybiB7c3RhcnRNczogdXRpbC5ub3coKSwgZW5kTXM6IG51bGx9O1xuICB9XG5cbiAgcHJpdmF0ZSBlbmRUaW1lcihxdWVyeTogV2ViR0xRdWVyeXxDUFVUaW1lclF1ZXJ5KTogV2ViR0xRdWVyeXxDUFVUaW1lclF1ZXJ5IHtcbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fUkVMSUFCTEUnKSA+IDApIHtcbiAgICAgIHRoaXMuZ3BncHUuZW5kUXVlcnkoKTtcbiAgICAgIHJldHVybiBxdWVyeTtcbiAgICB9XG4gICAgKHF1ZXJ5IGFzIENQVVRpbWVyUXVlcnkpLmVuZE1zID0gdXRpbC5ub3coKTtcbiAgICByZXR1cm4gcXVlcnk7XG4gIH1cblxuICBwcml2YXRlIGFzeW5jIGdldFF1ZXJ5VGltZShxdWVyeTogV2ViR0xRdWVyeXxDUFVUaW1lclF1ZXJ5KTogUHJvbWlzZTxudW1iZXI+IHtcbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fUkVMSUFCTEUnKSA+IDApIHtcbiAgICAgIHJldHVybiB0aGlzLmdwZ3B1LndhaXRGb3JRdWVyeUFuZEdldFRpbWUocXVlcnkgYXMgV2ViR0xRdWVyeSk7XG4gICAgfVxuICAgIGNvbnN0IHRpbWVyUXVlcnkgPSBxdWVyeSBhcyBDUFVUaW1lclF1ZXJ5O1xuICAgIHJldHVybiB0aW1lclF1ZXJ5LmVuZE1zIC0gdGltZXJRdWVyeS5zdGFydE1zO1xuICB9XG5cbiAgcHJpdmF0ZSBwZW5kaW5nRGVsZXRlcyA9IDA7XG5cbiAgLyoqXG4gICAqIERlY3JlYXNlIHRoZSBSZWZDb3VudCBvbiB0aGUgZGF0YUlkIGFuZCBkaXNwb3NlIHRoZSBtZW1vcnkgaWYgdGhlIGRhdGFJZFxuICAgKiBoYXMgMCByZWZDb3VudC4gSWYgdGhlcmUgYXJlIHBlbmRpbmcgcmVhZCBvbiB0aGUgZGF0YSwgdGhlIGRpc3Bvc2FsIHdvdWxkXG4gICAqIGFkZGVkIHRvIHRoZSBwZW5kaW5nIGRlbGV0ZSBxdWV1ZS4gUmV0dXJuIHRydWUgaWYgdGhlIGRhdGFJZCBpcyByZW1vdmVkXG4gICAqIGZyb20gYmFja2VuZCBvciB0aGUgYmFja2VuZCBkb2VzIG5vdCBjb250YWluIHRoZSBkYXRhSWQsIGZhbHNlIGlmIHRoZVxuICAgKiBkYXRhSWQgaXMgbm90IHJlbW92ZWQuIE1lbW9yeSBtYXkgb3IgbWF5IG5vdCBiZSByZWxlYXNlZCBldmVuIHdoZW4gZGF0YUlkXG4gICAqIGlzIHJlbW92ZWQsIHdoaWNoIGFsc28gZGVwZW5kcyBvbiBkYXRhUmVmQ291bnQsIHNlZSBgcmVsZWFzZUdQVWAuXG4gICAqIEBwYXJhbSBkYXRhSWRcbiAgICogQG9hcmFtIGZvcmNlIE9wdGlvbmFsLCByZW1vdmUgdGhlIGRhdGEgcmVnYXJkbGVzcyBvZiByZWZDb3VudFxuICAgKi9cbiAgb3ZlcnJpZGUgZGlzcG9zZURhdGEoZGF0YUlkOiBEYXRhSWQsIGZvcmNlID0gZmFsc2UpOiBib29sZWFuIHtcbiAgICBpZiAodGhpcy5wZW5kaW5nRGlzcG9zYWwuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG5cbiAgICAvLyBOby1vcCBpZiBhbHJlYWR5IGRpc3Bvc2VkLlxuICAgIGlmICghdGhpcy50ZXhEYXRhLmhhcyhkYXRhSWQpKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG5cbiAgICAvLyBpZiBmb3JjZSBmbGFnIGlzIHNldCwgY2hhbmdlIHJlZkNvdW50IHRvIDAsIHRoaXMgd291bGQgZW5zdXJlIGRpc3Bvc2FsXG4gICAgLy8gd2hlbiBhZGRlZCB0byB0aGUgcGVuZGluZ0Rpc3Bvc2FsIHF1ZXVlLiBNZW1vcnkgbWF5IG9yIG1heSBub3QgYmVcbiAgICAvLyByZWxlYXNlZCwgd2hpY2ggYWxzbyBkZXBlbmRzIG9uIGRhdGFSZWZDb3VudCwgc2VlIGByZWxlYXNlR1BVYC5cbiAgICBpZiAoZm9yY2UpIHtcbiAgICAgIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKS5yZWZDb3VudCA9IDA7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKS5yZWZDb3VudC0tO1xuICAgIH1cblxuICAgIGlmICghZm9yY2UgJiYgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnJlZkNvdW50ID4gMCkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnBlbmRpbmdSZWFkLmhhcyhkYXRhSWQpKSB7XG4gICAgICB0aGlzLnBlbmRpbmdEaXNwb3NhbC5hZGQoZGF0YUlkKTtcbiAgICAgIHRoaXMucGVuZGluZ0RlbGV0ZXMrKztcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG5cbiAgICB0aGlzLnJlbGVhc2VHUFVEYXRhKGRhdGFJZCk7XG4gICAgY29uc3Qge2NvbXBsZXhUZW5zb3JJbmZvc30gPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgaWYgKGNvbXBsZXhUZW5zb3JJbmZvcyAhPSBudWxsKSB7XG4gICAgICB0aGlzLmRpc3Bvc2VEYXRhKGNvbXBsZXhUZW5zb3JJbmZvcy5yZWFsLmRhdGFJZCwgZm9yY2UpO1xuICAgICAgdGhpcy5kaXNwb3NlRGF0YShjb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQsIGZvcmNlKTtcbiAgICB9XG5cbiAgICB0aGlzLnRleERhdGEuZGVsZXRlKGRhdGFJZCk7XG5cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIHByaXZhdGUgcmVsZWFzZUdQVURhdGEoZGF0YUlkOiBEYXRhSWQpOiB2b2lkIHtcbiAgICBjb25zdCB7dGV4dHVyZSwgZHR5cGUsIHRleFNoYXBlLCB1c2FnZSwgaXNQYWNrZWQsIHNsaWNlfSA9XG4gICAgICAgIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCBrZXkgPSBzbGljZSAmJiBzbGljZS5vcmlnRGF0YUlkIHx8IGRhdGFJZDtcbiAgICBjb25zdCByZWZDb3VudCA9IHRoaXMuZGF0YVJlZkNvdW50LmdldChrZXkpO1xuXG4gICAgaWYgKHJlZkNvdW50ID4gMSkge1xuICAgICAgdGhpcy5kYXRhUmVmQ291bnQuc2V0KGtleSwgcmVmQ291bnQgLSAxKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5kYXRhUmVmQ291bnQuZGVsZXRlKGtleSk7XG4gICAgICBpZiAodGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICAgIHRoaXMubnVtQnl0ZXNJbkdQVSAtPSB0aGlzLmNvbXB1dGVCeXRlcyh0ZXhTaGFwZSwgZHR5cGUpO1xuICAgICAgICB0aGlzLnRleHR1cmVNYW5hZ2VyLnJlbGVhc2VUZXh0dXJlKHRleHR1cmUsIHRleFNoYXBlLCB1c2FnZSwgaXNQYWNrZWQpO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgdGV4RGF0YS50ZXh0dXJlID0gbnVsbDtcbiAgICB0ZXhEYXRhLnRleFNoYXBlID0gbnVsbDtcbiAgICB0ZXhEYXRhLmlzUGFja2VkID0gZmFsc2U7XG4gICAgdGV4RGF0YS5zbGljZSA9IG51bGw7XG4gIH1cblxuICBnZXRUZXh0dXJlKGRhdGFJZDogRGF0YUlkKTogV2ViR0xUZXh0dXJlIHtcbiAgICB0aGlzLnVwbG9hZFRvR1BVKGRhdGFJZCk7XG4gICAgcmV0dXJuIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKS50ZXh0dXJlLnRleHR1cmU7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBpbnRlcm5hbCBpbmZvcm1hdGlvbiBmb3IgdGhlIHNwZWNpZmljIGRhdGEgYnVja2V0LiBVc2VkIGluIHVuaXRcbiAgICogdGVzdHMuXG4gICAqL1xuICBnZXREYXRhSW5mbyhkYXRhSWQ6IERhdGFJZCk6IFRleHR1cmVEYXRhIHtcbiAgICByZXR1cm4gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICB9XG5cbiAgLypcbiAgVGVzdHMgd2hldGhlciBhbGwgdGhlIGlucHV0cyB0byBhbiBvcCBhcmUgc21hbGwgYW5kIG9uIHRoZSBDUFUuIFRoaXMgaGV1cmlzdGljXG4gIGRldGVybWluZXMgd2hlbiBpdCB3b3VsZCBiZSBmYXN0ZXIgdG8gZXhlY3V0ZSBhIGtlcm5lbCBvbiB0aGUgQ1BVLiBXZWJHTFxuICBrZXJuZWxzIG9wdCBpbnRvIHJ1bm5pbmcgdGhpcyBjaGVjayBhbmQgZm9yd2FyZGluZyB3aGVuIGFwcHJvcHJpYXRlLlxuICBUT0RPKGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzg3Mik6IERldmVsb3AgYSBtb3JlXG4gIHN1c3RhaW5hYmxlIHN0cmF0ZWd5IGZvciBvcHRpbWl6aW5nIGJhY2tlbmQgZXhlY3V0aW9uIG9mIG9wcy5cbiAgICovXG4gIHNob3VsZEV4ZWN1dGVPbkNQVShcbiAgICAgIGlucHV0czogVGVuc29ySW5mb1tdLFxuICAgICAgc2l6ZVRocmVzaG9sZCA9IENQVV9IQU5ET0ZGX1NJWkVfVEhSRVNIT0xEKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGVudigpLmdldEJvb2woJ1dFQkdMX0NQVV9GT1JXQVJEJykgJiZcbiAgICAgICAgaW5wdXRzLmV2ZXJ5KFxuICAgICAgICAgICAgaW5wdXQgPT4gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpLnRleHR1cmUgPT0gbnVsbCAmJlxuICAgICAgICAgICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZShpbnB1dC5zaGFwZSkgPCBzaXplVGhyZXNob2xkKTtcbiAgfVxuXG4gIGdldEdQR1BVQ29udGV4dCgpOiBHUEdQVUNvbnRleHQge1xuICAgIHJldHVybiB0aGlzLmdwZ3B1O1xuICB9XG5cbiAgd2hlcmUoY29uZGl0aW9uOiBUZW5zb3IpOiBUZW5zb3IyRCB7XG4gICAgYmFja2VuZF91dGlsLndhcm4oXG4gICAgICAgICd0Zi53aGVyZSgpIGluIHdlYmdsIGxvY2tzIHRoZSBVSSB0aHJlYWQuICcgK1xuICAgICAgICAnQ2FsbCB0Zi53aGVyZUFzeW5jKCkgaW5zdGVhZCcpO1xuICAgIGNvbnN0IGNvbmRWYWxzID0gY29uZGl0aW9uLmRhdGFTeW5jKCk7XG4gICAgcmV0dXJuIHdoZXJlSW1wbChjb25kaXRpb24uc2hhcGUsIGNvbmRWYWxzKTtcbiAgfVxuXG4gIHByaXZhdGUgcGFja2VkVW5hcnlPcCh4OiBUZW5zb3JJbmZvLCBvcDogc3RyaW5nLCBkdHlwZTogRGF0YVR5cGUpIHtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFVuYXJ5T3BQYWNrZWRQcm9ncmFtKHguc2hhcGUsIG9wKTtcbiAgICBjb25zdCBvdXRJbmZvID0gdGhpcy5jb21waWxlQW5kUnVuKHByb2dyYW0sIFt4XSwgZHR5cGUpO1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbVRlbnNvckluZm8ob3V0SW5mbyk7XG4gIH1cblxuICAvLyBUT0RPKG1zb3VsYW5pbGxlKSByZW1vdmUgdGhpcyBvbmNlIHRoZSBiYWNrZW5kIGhhcyBiZWVuIG1vZHVsYXJpemVkXG4gIC8vIGEgY29weSBpcyBuZWVkZWQgaGVyZSB0byBicmVhayBhIGNpcmN1bGFyIGRlcGVuZGVuY3kuXG4gIC8vIEFsc28gcmVtb3ZlIHRoZSBvcCBmcm9tIHVuYXJ5X29wLlxuICBhYnM8VCBleHRlbmRzIFRlbnNvcj4oeDogVCk6IFQge1xuICAgIC8vIFRPRE86IGhhbmRsZSBjYXNlcyB3aGVuIHggaXMgY29tcGxleC5cbiAgICBpZiAodGhpcy5zaG91bGRFeGVjdXRlT25DUFUoW3hdKSAmJiB4LmR0eXBlICE9PSAnY29tcGxleDY0Jykge1xuICAgICAgY29uc3Qgb3V0VmFsdWVzID1cbiAgICAgICAgICBzaW1wbGVBYnNJbXBsQ1BVKHRoaXMudGV4RGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5KTtcbiAgICAgIHJldHVybiB0aGlzLm1ha2VPdXRwdXQoeC5zaGFwZSwgeC5kdHlwZSwgb3V0VmFsdWVzKTtcbiAgICB9XG5cbiAgICBpZiAoZW52KCkuZ2V0Qm9vbCgnV0VCR0xfUEFDS19VTkFSWV9PUEVSQVRJT05TJykpIHtcbiAgICAgIHJldHVybiB0aGlzLnBhY2tlZFVuYXJ5T3AoeCwgdW5hcnlfb3AuQUJTLCB4LmR0eXBlKSBhcyBUO1xuICAgIH1cblxuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oeC5zaGFwZSwgdW5hcnlfb3AuQUJTKTtcbiAgICBjb25zdCBvdXRJbmZvID0gdGhpcy5jb21waWxlQW5kUnVuKHByb2dyYW0sIFt4XSk7XG4gICAgcmV0dXJuIGVuZ2luZSgpLm1ha2VUZW5zb3JGcm9tVGVuc29ySW5mbyhvdXRJbmZvKSBhcyBUO1xuICB9XG5cbiAgbWFrZVRlbnNvckluZm8oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIHZhbHVlcz86IEJhY2tlbmRWYWx1ZXN8c3RyaW5nW10pOiBUZW5zb3JJbmZvIHtcbiAgICBsZXQgZGF0YUlkO1xuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycgJiYgdmFsdWVzICE9IG51bGwgJiYgdmFsdWVzLmxlbmd0aCA+IDAgJiZcbiAgICAgICAgdXRpbC5pc1N0cmluZyh2YWx1ZXNbMF0pKSB7XG4gICAgICBjb25zdCBlbmNvZGVkVmFsdWVzID1cbiAgICAgICAgICAodmFsdWVzIGFzIHVua25vd24gYXMgc3RyaW5nW10pLm1hcChkID0+IHV0aWwuZW5jb2RlU3RyaW5nKGQpKTtcblxuICAgICAgZGF0YUlkID0gdGhpcy53cml0ZShlbmNvZGVkVmFsdWVzLCBzaGFwZSwgZHR5cGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICBkYXRhSWQgPSB0aGlzLndyaXRlKHZhbHVlcyBhcyBUeXBlZEFycmF5LCBzaGFwZSwgZHR5cGUpO1xuICAgIH1cblxuICAgIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKS51c2FnZSA9IG51bGw7XG4gICAgcmV0dXJuIHtkYXRhSWQsIHNoYXBlLCBkdHlwZX07XG4gIH1cblxuICBwcml2YXRlIG1ha2VPdXRwdXQ8VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSwgdmFsdWVzPzogQmFja2VuZFZhbHVlcyk6IFQge1xuICAgIHJldHVybiBlbmdpbmUoKS5tYWtlVGVuc29yRnJvbVRlbnNvckluZm8oXG4gICAgICAgICAgICAgICB0aGlzLm1ha2VUZW5zb3JJbmZvKHNoYXBlLCBkdHlwZSwgdmFsdWVzKSwgdGhpcykgYXMgVDtcbiAgfVxuXG4gIHVucGFja1RlbnNvcihpbnB1dDogVGVuc29ySW5mbyk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5wYWNrUHJvZ3JhbShpbnB1dC5zaGFwZSk7XG4gICAgcmV0dXJuIHRoaXMucnVuV2ViR0xQcm9ncmFtKHByb2dyYW0sIFtpbnB1dF0sIGlucHV0LmR0eXBlKTtcbiAgfVxuXG4gIHBhY2tUZW5zb3IoaW5wdXQ6IFRlbnNvckluZm8pOiBUZW5zb3JJbmZvIHtcbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFBhY2tQcm9ncmFtKGlucHV0LnNoYXBlKTtcbiAgICBjb25zdCBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPdXRwdXQgPSB0cnVlO1xuICAgIHJldHVybiB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW2lucHV0XSwgaW5wdXQuZHR5cGUsIG51bGwgLyogY3VzdG9tVW5pZm9ybVZhbHVlcyAqLyxcbiAgICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nT3V0cHV0KTtcbiAgfVxuXG4gIHByaXZhdGUgcGFja2VkUmVzaGFwZShpbnB1dDogVGVuc29ySW5mbywgYWZ0ZXJTaGFwZTogbnVtYmVyW10pOiBUZW5zb3JJbmZvIHtcbiAgICBjb25zdCBpbnB1dDNEU2hhcGUgPSBbXG4gICAgICB3ZWJnbF91dGlsLmdldEJhdGNoRGltKGlucHV0LnNoYXBlKSxcbiAgICAgIC4uLndlYmdsX3V0aWwuZ2V0Um93c0NvbHMoaW5wdXQuc2hhcGUpXG4gICAgXSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gICAgY29uc3QgaW5wdXQzRDogVGVuc29ySW5mbyA9IHtcbiAgICAgIGR0eXBlOiBpbnB1dC5kdHlwZSxcbiAgICAgIHNoYXBlOiBpbnB1dDNEU2hhcGUsXG4gICAgICBkYXRhSWQ6IGlucHV0LmRhdGFJZFxuICAgIH07XG4gICAgY29uc3QgYWZ0ZXJTaGFwZUFzM0QgPSBbXG4gICAgICB3ZWJnbF91dGlsLmdldEJhdGNoRGltKGFmdGVyU2hhcGUpLCAuLi53ZWJnbF91dGlsLmdldFJvd3NDb2xzKGFmdGVyU2hhcGUpXG4gICAgXSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG5cbiAgICBjb25zdCBwcm9ncmFtID0gbmV3IFJlc2hhcGVQYWNrZWRQcm9ncmFtKGFmdGVyU2hhcGVBczNELCBpbnB1dDNEU2hhcGUpO1xuICAgIGNvbnN0IHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0ID0gdHJ1ZTtcbiAgICBjb25zdCBjdXN0b21WYWx1ZXMgPSBbaW5wdXQzRFNoYXBlXTtcbiAgICBjb25zdCBvdXRwdXQgPSB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW2lucHV0M0RdLCBpbnB1dC5kdHlwZSwgY3VzdG9tVmFsdWVzLFxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCk7XG4gICAgcmV0dXJuIHtkYXRhSWQ6IG91dHB1dC5kYXRhSWQsIHNoYXBlOiBhZnRlclNoYXBlLCBkdHlwZTogb3V0cHV0LmR0eXBlfTtcbiAgfVxuXG4gIHByaXZhdGUgZGVjb2RlKGRhdGFJZDogRGF0YUlkLCBjdXN0b21UZXhTaGFwZT86IFtudW1iZXIsIG51bWJlcl0pOlxuICAgICAgVGVuc29ySW5mbyB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7aXNQYWNrZWQsIHNoYXBlLCBkdHlwZX0gPSB0ZXhEYXRhO1xuICAgIGlmIChjdXN0b21UZXhTaGFwZSAhPSBudWxsKSB7XG4gICAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICAgIGNvbnN0IHRleFNpemUgPSBjdXN0b21UZXhTaGFwZVswXSAqIGN1c3RvbVRleFNoYXBlWzFdICogNDtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIHNpemUgPD0gdGV4U2l6ZSxcbiAgICAgICAgICAoKSA9PiAnY3VzdG9tVGV4U2hhcGUgaXMgdG9vIHNtYWxsLiAnICtcbiAgICAgICAgICAgICAgJ1JvdyAqIENvbHVtbiAqIDQgc2hvdWxkIGJlIGVxdWFsIG9yIGxhcmdlciB0aGFuIHRoZSAnICtcbiAgICAgICAgICAgICAgJ3NpemUgb2YgdGhlIHRlbnNvciBkYXRhLicpO1xuICAgIH1cbiAgICBjb25zdCBzaGFwZUFzM0QgPVxuICAgICAgICB3ZWJnbF91dGlsLmdldFNoYXBlQXMzRChzaGFwZSkgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICAgIGxldCBwcm9ncmFtO1xuICAgIGlmIChpc1BhY2tlZCkge1xuICAgICAgcHJvZ3JhbSA9IG5ldyBEZWNvZGVNYXRyaXhQYWNrZWRQcm9ncmFtKHNoYXBlQXMzRCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHByb2dyYW0gPSBuZXcgRGVjb2RlTWF0cml4UHJvZ3JhbShzaGFwZUFzM0QpO1xuICAgIH1cbiAgICBjb25zdCBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9IHRydWU7XG4gICAgY29uc3QgY3VzdG9tVmFsdWVzID1cbiAgICAgICAgW2N1c3RvbVRleFNoYXBlICE9IG51bGwgPyBjdXN0b21UZXhTaGFwZSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGV4X3V0aWwuZ2V0RGVuc2VUZXhTaGFwZShzaGFwZUFzM0QpXTtcbiAgICBjb25zdCBvdXQgPSB0aGlzLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgICAgcHJvZ3JhbSwgW3tzaGFwZTogc2hhcGVBczNELCBkdHlwZSwgZGF0YUlkfV0sIGR0eXBlLCBjdXN0b21WYWx1ZXMsXG4gICAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0LCBjdXN0b21UZXhTaGFwZSk7XG4gICAgcmV0dXJuIHtkdHlwZSwgc2hhcGUsIGRhdGFJZDogb3V0LmRhdGFJZH07XG4gIH1cblxuICBydW5XZWJHTFByb2dyYW0oXG4gICAgICBwcm9ncmFtOiBHUEdQVVByb2dyYW0sIGlucHV0czogVGVuc29ySW5mb1tdLCBvdXRwdXREdHlwZTogRGF0YVR5cGUsXG4gICAgICBjdXN0b21Vbmlmb3JtVmFsdWVzPzogbnVtYmVyW11bXSwgcHJldmVudEVhZ2VyVW5wYWNraW5nT2ZPdXRwdXQgPSBmYWxzZSxcbiAgICAgIGN1c3RvbVRleFNoYXBlPzogW251bWJlciwgbnVtYmVyXSk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IG91dHB1dCA9IHRoaXMubWFrZVRlbnNvckluZm8ocHJvZ3JhbS5vdXRwdXRTaGFwZSwgb3V0cHV0RHR5cGUpO1xuICAgIGNvbnN0IG91dERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KG91dHB1dC5kYXRhSWQpO1xuICAgIGlmIChwcm9ncmFtLnBhY2tlZE91dHB1dCkge1xuICAgICAgb3V0RGF0YS5pc1BhY2tlZCA9IHRydWU7XG4gICAgfVxuICAgIGlmIChwcm9ncmFtLm91dFBhY2tpbmdTY2hlbWUgPT09IHRleF91dGlsLlBhY2tpbmdTY2hlbWUuREVOU0UpIHtcbiAgICAgIGNvbnN0IHRleGVsU2hhcGUgPSBjdXN0b21UZXhTaGFwZSAhPSBudWxsID9cbiAgICAgICAgICBjdXN0b21UZXhTaGFwZSA6XG4gICAgICAgICAgdGV4X3V0aWwuZ2V0RGVuc2VUZXhTaGFwZShwcm9ncmFtLm91dHB1dFNoYXBlKTtcbiAgICAgIC8vIEZvciBhIGRlbnNlbHkgcGFja2VkIG91dHB1dCwgd2UgZXhwbGljaXRseSBzZXQgdGV4U2hhcGVcbiAgICAgIC8vIHNvIGl0IGRvZXNuJ3QgZ2V0IGFzc2lnbmVkIGxhdGVyIGFjY29yZGluZyB0byBvdXIgdHlwaWNhbCBwYWNraW5nXG4gICAgICAvLyBzY2hlbWUgd2hlcmVpbiBhIHNpbmdsZSB0ZXhlbCBjYW4gb25seSBjb250YWluIHZhbHVlcyBmcm9tIGFkamFjZW50XG4gICAgICAvLyByb3dzL2NvbHMuXG4gICAgICBvdXREYXRhLnRleFNoYXBlID0gdGV4ZWxTaGFwZS5tYXAoZCA9PiBkICogMikgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0ub3V0VGV4VXNhZ2UgIT0gbnVsbCkge1xuICAgICAgb3V0RGF0YS51c2FnZSA9IHByb2dyYW0ub3V0VGV4VXNhZ2U7XG4gICAgfVxuXG4gICAgaWYgKHV0aWwuc2l6ZUZyb21TaGFwZShvdXRwdXQuc2hhcGUpID09PSAwKSB7XG4gICAgICAvLyBTaG9ydC1jaXJjdWl0IHRoZSBjb21wdXRhdGlvbiBzaW5jZSB0aGUgcmVzdWx0IGlzIGVtcHR5IChoYXMgMCBpbiBpdHNcbiAgICAgIC8vIHNoYXBlKS5cbiAgICAgIG91dERhdGEudmFsdWVzID1cbiAgICAgICAgICB1dGlsLmdldFR5cGVkQXJyYXlGcm9tRFR5cGUob3V0cHV0LmR0eXBlIGFzICdmbG9hdDMyJywgMCk7XG4gICAgICByZXR1cm4gb3V0cHV0O1xuICAgIH1cblxuICAgIGNvbnN0IGRhdGFUb0Rpc3Bvc2U6IFRlbnNvckluZm9bXSA9IFtdO1xuICAgIGNvbnN0IGlucHV0c0RhdGE6IFRlbnNvckRhdGFbXSA9IGlucHV0cy5tYXAoaW5wdXQgPT4ge1xuICAgICAgaWYgKGlucHV0LmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgR1BHUFVQcm9ncmFtIGRvZXMgbm90IHN1cHBvcnQgY29tcGxleDY0IGlucHV0LiBGb3IgY29tcGxleDY0IGAgK1xuICAgICAgICAgICAgYGR0eXBlcywgcGxlYXNlIHNlcGFyYXRlIHRoZSBwcm9ncmFtIGludG8gcmVhbCBhbmQgaW1hZ2luYXJ5IGAgK1xuICAgICAgICAgICAgYHBhcnRzLmApO1xuICAgICAgfVxuXG4gICAgICBsZXQgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoaW5wdXQuZGF0YUlkKTtcblxuICAgICAgaWYgKHRleERhdGEudGV4dHVyZSA9PSBudWxsKSB7XG4gICAgICAgIGlmICghcHJvZ3JhbS5wYWNrZWRJbnB1dHMgJiZcbiAgICAgICAgICAgIHV0aWwuc2l6ZUZyb21TaGFwZShpbnB1dC5zaGFwZSkgPD1cbiAgICAgICAgICAgICAgICBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1NJWkVfVVBMT0FEX1VOSUZPUk0nKSkge1xuICAgICAgICAgIC8vIFVwbG9hZCBzbWFsbCB0ZW5zb3JzIHRoYXQgbGl2ZSBvbiB0aGUgQ1BVIGFzIHVuaWZvcm1zLCBub3QgYXNcbiAgICAgICAgICAvLyB0ZXh0dXJlcy4gRG8gdGhpcyBvbmx5IHdoZW4gdGhlIGVudmlyb25tZW50IHN1cHBvcnRzIDMyYml0IGZsb2F0c1xuICAgICAgICAgIC8vIGR1ZSB0byBwcm9ibGVtcyB3aGVuIGNvbXBhcmluZyAxNmJpdCBmbG9hdHMgd2l0aCAzMmJpdCBmbG9hdHMuXG4gICAgICAgICAgLy8gVE9ETyhodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2lzc3Vlcy84MjEpOiBNYWtlIGl0XG4gICAgICAgICAgLy8gcG9zc2libGUgZm9yIHBhY2tlZCBzaGFkZXJzIHRvIHNhbXBsZSBmcm9tIHVuaWZvcm1zLlxuICAgICAgICAgIHJldHVybiB7XG4gICAgICAgICAgICBzaGFwZTogaW5wdXQuc2hhcGUsXG4gICAgICAgICAgICB0ZXhEYXRhOiBudWxsLFxuICAgICAgICAgICAgaXNVbmlmb3JtOiB0cnVlLFxuICAgICAgICAgICAgdW5pZm9ybVZhbHVlczogdGV4RGF0YS52YWx1ZXMgYXMgVHlwZWRBcnJheVxuICAgICAgICAgIH07XG4gICAgICAgIH1cblxuICAgICAgICAvLyBUaGlzIGVuc3VyZXMgdGhhdCBpZiBhIHBhY2tlZCBwcm9ncmFtJ3MgaW5wdXRzIGhhdmUgbm90IHlldCBiZWVuXG4gICAgICAgIC8vIHVwbG9hZGVkIHRvIHRoZSBHUFUsIHRoZXkgZ2V0IHVwbG9hZGVkIGFzIHBhY2tlZCByaWdodCBvZmYgdGhlIGJhdC5cbiAgICAgICAgaWYgKHByb2dyYW0ucGFja2VkSW5wdXRzKSB7XG4gICAgICAgICAgdGV4RGF0YS5pc1BhY2tlZCA9IHRydWU7XG4gICAgICAgICAgdGV4RGF0YS5zaGFwZSA9IGlucHV0LnNoYXBlO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHRoaXMudXBsb2FkVG9HUFUoaW5wdXQuZGF0YUlkKTtcbiAgICAgIGlmICghIXRleERhdGEuaXNQYWNrZWQgIT09ICEhcHJvZ3JhbS5wYWNrZWRJbnB1dHMpIHtcbiAgICAgICAgaW5wdXQgPSB0ZXhEYXRhLmlzUGFja2VkID8gdGhpcy51bnBhY2tUZW5zb3IoaW5wdXQpIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy5wYWNrVGVuc29yKGlucHV0KTtcbiAgICAgICAgZGF0YVRvRGlzcG9zZS5wdXNoKGlucHV0KTtcbiAgICAgICAgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoaW5wdXQuZGF0YUlkKTtcbiAgICAgIH0gZWxzZSBpZiAoXG4gICAgICAgICAgdGV4RGF0YS5pc1BhY2tlZCAmJlxuICAgICAgICAgICF3ZWJnbF91dGlsLmlzUmVzaGFwZUZyZWUodGV4RGF0YS5zaGFwZSwgaW5wdXQuc2hhcGUpKSB7XG4gICAgICAgIC8vIFRoaXMgaXMgYSBzcGVjaWFsIGNhc2Ugd2hlcmUgYSB0ZXh0dXJlIGV4aXN0cyBmb3IgYSB0ZW5zb3JcbiAgICAgICAgLy8gYnV0IHRoZSBzaGFwZXMgYXJlIGluY29tcGF0aWJsZSAoZHVlIHRvIHBhY2tpbmcgY29uc3RyYWludHMpIGJlY2F1c2VcbiAgICAgICAgLy8gdGhlIHRlbnNvciBkaWQgbm90IGhhdmUgYSBjaGFuY2UgdG8gZ28gdGhyb3VnaCB0aGUgcGFja2VkIHJlc2hhcGVcbiAgICAgICAgLy8gc2hhZGVyLiBUaGlzIG9ubHkgaGFwcGVucyB3aGVuIHdlIHJlc2hhcGUgdGhlICpzYW1lKiB0ZW5zb3IgdG8gZm9ybVxuICAgICAgICAvLyAqZGlzdGluY3QqIGlucHV0cyB0byBhbiBvcCwgZS5nLiBkb3R0aW5nIGEgdmVjdG9yIHdpdGggaXRzZWxmLiBUaGlzXG4gICAgICAgIC8vIGNhc2Ugd2lsbCBkaXNhcHBlYXIgb25jZSBwYWNrZWQgdXBsb2FkaW5nIGlzIHRoZSBkZWZhdWx0LlxuXG4gICAgICAgIGNvbnN0IHNhdmVkSW5wdXQgPSBpbnB1dDtcbiAgICAgICAgY29uc3QgdGFyZ2V0U2hhcGUgPSBpbnB1dC5zaGFwZTtcblxuICAgICAgICBpbnB1dC5zaGFwZSA9IHRleERhdGEuc2hhcGU7XG4gICAgICAgIGlucHV0ID0gdGhpcy5wYWNrZWRSZXNoYXBlKGlucHV0IGFzIFRlbnNvciwgdGFyZ2V0U2hhcGUpO1xuICAgICAgICBkYXRhVG9EaXNwb3NlLnB1c2goaW5wdXQpO1xuICAgICAgICB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpO1xuXG4gICAgICAgIHNhdmVkSW5wdXQuc2hhcGUgPSB0YXJnZXRTaGFwZTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIHtzaGFwZTogaW5wdXQuc2hhcGUsIHRleERhdGEsIGlzVW5pZm9ybTogZmFsc2V9O1xuICAgIH0pO1xuXG4gICAgdGhpcy51cGxvYWRUb0dQVShvdXRwdXQuZGF0YUlkKTtcbiAgICBjb25zdCBvdXRwdXREYXRhOlxuICAgICAgICBUZW5zb3JEYXRhID0ge3NoYXBlOiBvdXRwdXQuc2hhcGUsIHRleERhdGE6IG91dERhdGEsIGlzVW5pZm9ybTogZmFsc2V9O1xuICAgIGNvbnN0IGtleSA9IGdwZ3B1X21hdGgubWFrZVNoYWRlcktleShwcm9ncmFtLCBpbnB1dHNEYXRhLCBvdXRwdXREYXRhKTtcbiAgICBjb25zdCBiaW5hcnkgPSB0aGlzLmdldEFuZFNhdmVCaW5hcnkoa2V5LCAoKSA9PiB7XG4gICAgICByZXR1cm4gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShcbiAgICAgICAgICB0aGlzLmdwZ3B1LCBwcm9ncmFtLCBpbnB1dHNEYXRhLCBvdXRwdXREYXRhKTtcbiAgICB9KTtcbiAgICBjb25zdCBzaG91bGRUaW1lUHJvZ3JhbSA9IHRoaXMuYWN0aXZlVGltZXJzICE9IG51bGw7XG4gICAgbGV0IHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnk7XG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICBxdWVyeSA9IHRoaXMuc3RhcnRUaW1lcigpO1xuICAgIH1cblxuICAgIGlmICghZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShcbiAgICAgICAgICB0aGlzLmdwZ3B1LCBiaW5hcnksIGlucHV0c0RhdGEsIG91dHB1dERhdGEsIGN1c3RvbVVuaWZvcm1WYWx1ZXMpO1xuICAgIH1cblxuICAgIGRhdGFUb0Rpc3Bvc2UuZm9yRWFjaChpbmZvID0+IHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oaW5mbykpO1xuXG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICBxdWVyeSA9IHRoaXMuZW5kVGltZXIocXVlcnkpO1xuICAgICAgdGhpcy5hY3RpdmVUaW1lcnMucHVzaChcbiAgICAgICAgICB7bmFtZTogcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lLCBxdWVyeTogdGhpcy5nZXRRdWVyeVRpbWUocXVlcnkpfSk7XG4gICAgfVxuXG4gICAgY29uc3QgZ2xGbHVzaFRocmVzaG9sZCA9IGVudigpLmdldCgnV0VCR0xfRkxVU0hfVEhSRVNIT0xEJyk7XG4gICAgLy8gTWFudWFsbHkgR0wgZmx1c2ggcmVxdWVzdGVkXG4gICAgaWYgKGdsRmx1c2hUaHJlc2hvbGQgPiAwKSB7XG4gICAgICBjb25zdCB0aW1lID0gdXRpbC5ub3coKTtcbiAgICAgIGlmICgodGltZSAtIHRoaXMubGFzdEdsRmx1c2hUaW1lKSA+IGdsRmx1c2hUaHJlc2hvbGQpIHtcbiAgICAgICAgdGhpcy5ncGdwdS5nbC5mbHVzaCgpO1xuICAgICAgICB0aGlzLmxhc3RHbEZsdXNoVGltZSA9IHRpbWU7XG4gICAgICB9XG4gICAgfVxuXG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdXRUJHTF9MQVpJTFlfVU5QQUNLJykgJiYgb3V0RGF0YS5pc1BhY2tlZCAmJlxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9PT0gZmFsc2UpIHtcbiAgICAgIGNvbnN0IHVucGFja2VkID0gdGhpcy51bnBhY2tUZW5zb3Iob3V0cHV0KTtcbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ob3V0cHV0KTtcbiAgICAgIHJldHVybiB1bnBhY2tlZDtcbiAgICB9XG4gICAgcmV0dXJuIG91dHB1dDtcbiAgfVxuXG4gIGNvbXBpbGVBbmRSdW4oXG4gICAgICBwcm9ncmFtOiBHUEdQVVByb2dyYW0sIGlucHV0czogVGVuc29ySW5mb1tdLCBvdXRwdXREdHlwZT86IERhdGFUeXBlLFxuICAgICAgY3VzdG9tVW5pZm9ybVZhbHVlcz86IG51bWJlcltdW10sXG4gICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9IGZhbHNlKTogVGVuc29ySW5mbyB7XG4gICAgb3V0cHV0RHR5cGUgPSBvdXRwdXREdHlwZSB8fCBpbnB1dHNbMF0uZHR5cGU7XG4gICAgY29uc3Qgb3V0SW5mbyA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBpbnB1dHMsIG91dHB1dER0eXBlLCBjdXN0b21Vbmlmb3JtVmFsdWVzLFxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCk7XG4gICAgcmV0dXJuIG91dEluZm87XG4gIH1cblxuICBwcml2YXRlIGdldEFuZFNhdmVCaW5hcnkoa2V5OiBzdHJpbmcsIGdldEJpbmFyeTogKCkgPT4gR1BHUFVCaW5hcnkpOlxuICAgICAgR1BHUFVCaW5hcnkge1xuICAgIGlmICghKGtleSBpbiB0aGlzLmJpbmFyeUNhY2hlKSkge1xuICAgICAgdGhpcy5iaW5hcnlDYWNoZVtrZXldID0gZ2V0QmluYXJ5KCk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmJpbmFyeUNhY2hlW2tleV07XG4gIH1cblxuICBnZXRUZXh0dXJlTWFuYWdlcigpOiBUZXh0dXJlTWFuYWdlciB7XG4gICAgcmV0dXJuIHRoaXMudGV4dHVyZU1hbmFnZXI7XG4gIH1cblxuICBwcml2YXRlIGRpc3Bvc2VkID0gZmFsc2U7XG5cbiAgb3ZlcnJpZGUgZGlzcG9zZSgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICAvLyBBdm9pZCBkaXNwb3NpbmcgdGhlIGNvbXBpbGVkIHdlYmdsIHByb2dyYW1zIGR1cmluZyB1bml0IHRlc3RpbmcgYmVjYXVzZVxuICAgIC8vIGl0IHNsb3dzIGRvd24gdGVzdCBleGVjdXRpb24uXG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdJU19URVNUJykpIHtcbiAgICAgIGNvbnN0IGFsbEtleXMgPSBPYmplY3Qua2V5cyh0aGlzLmJpbmFyeUNhY2hlKTtcbiAgICAgIGFsbEtleXMuZm9yRWFjaChrZXkgPT4ge1xuICAgICAgICB0aGlzLmdwZ3B1LmRlbGV0ZVByb2dyYW0odGhpcy5iaW5hcnlDYWNoZVtrZXldLndlYkdMUHJvZ3JhbSk7XG4gICAgICAgIGRlbGV0ZSB0aGlzLmJpbmFyeUNhY2hlW2tleV07XG4gICAgICB9KTtcbiAgICB9XG4gICAgdGhpcy50ZXh0dXJlTWFuYWdlci5kaXNwb3NlKCk7XG4gICAgaWYgKHRoaXMuY2FudmFzICE9IG51bGwgJiZcbiAgICAgICAgKHR5cGVvZiAoSFRNTENhbnZhc0VsZW1lbnQpICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgICAgdGhpcy5jYW52YXMgaW5zdGFuY2VvZiBIVE1MQ2FudmFzRWxlbWVudCkpIHtcbiAgICAgIHRoaXMuY2FudmFzLnJlbW92ZSgpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNhbnZhcyA9IG51bGw7XG4gICAgfVxuICAgIGlmICh0aGlzLmdwZ3B1Q3JlYXRlZExvY2FsbHkpIHtcbiAgICAgIHRoaXMuZ3BncHUucHJvZ3JhbSA9IG51bGw7XG4gICAgICB0aGlzLmdwZ3B1LmRpc3Bvc2UoKTtcbiAgICB9XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBvdmVycmlkZSBmbG9hdFByZWNpc2lvbigpOiAxNnwzMiB7XG4gICAgaWYgKHRoaXMuZmxvYXRQcmVjaXNpb25WYWx1ZSA9PSBudWxsKSB7XG4gICAgICB0aGlzLmZsb2F0UHJlY2lzaW9uVmFsdWUgPSB0aWR5KCgpID0+IHtcbiAgICAgICAgaWYgKCFlbnYoKS5nZXQoJ1dFQkdMX1JFTkRFUl9GTE9BVDMyX0VOQUJMRUQnKSkge1xuICAgICAgICAgIC8vIE1vbWVudGFyaWx5IHN3aXRjaGluZyBERUJVRyBmbGFnIHRvIGZhbHNlIHNvIHdlIGRvbid0IHRocm93IGFuXG4gICAgICAgICAgLy8gZXJyb3IgdHJ5aW5nIHRvIHVwbG9hZCBhIHNtYWxsIHZhbHVlLlxuICAgICAgICAgIGNvbnN0IGRlYnVnRmxhZyA9IGVudigpLmdldEJvb2woJ0RFQlVHJyk7XG4gICAgICAgICAgZW52KCkuc2V0KCdERUJVRycsIGZhbHNlKTtcbiAgICAgICAgICBjb25zdCB1bmRlcmZsb3dDaGVja1ZhbHVlID0gdGhpcy5hYnMoc2NhbGFyKDFlLTgpKS5kYXRhU3luYygpWzBdO1xuICAgICAgICAgIGVudigpLnNldCgnREVCVUcnLCBkZWJ1Z0ZsYWcpO1xuXG4gICAgICAgICAgaWYgKHVuZGVyZmxvd0NoZWNrVmFsdWUgPiAwKSB7XG4gICAgICAgICAgICByZXR1cm4gMzI7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiAxNjtcbiAgICAgIH0pO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5mbG9hdFByZWNpc2lvblZhbHVlO1xuICB9XG5cbiAgLyoqIFJldHVybnMgdGhlIHNtYWxsZXN0IHJlcHJlc2VudGFibGUgbnVtYmVyLiAgKi9cbiAgb3ZlcnJpZGUgZXBzaWxvbigpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmZsb2F0UHJlY2lzaW9uKCkgPT09IDMyID8gRVBTSUxPTl9GTE9BVDMyIDogRVBTSUxPTl9GTE9BVDE2O1xuICB9XG5cbiAgdXBsb2FkVG9HUFUoZGF0YUlkOiBEYXRhSWQpOiB2b2lkIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHtzaGFwZSwgZHR5cGUsIHZhbHVlcywgdGV4dHVyZSwgdXNhZ2UsIGlzUGFja2VkfSA9IHRleERhdGE7XG5cbiAgICBpZiAodGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICAvLyBBcnJheSBpcyBhbHJlYWR5IG9uIEdQVS4gTm8tb3AuXG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IHNob3VsZFRpbWVQcm9ncmFtID0gdGhpcy5hY3RpdmVUaW1lcnMgIT0gbnVsbDtcbiAgICBsZXQgc3RhcnQ6IG51bWJlcjtcbiAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgIHN0YXJ0ID0gdXRpbC5ub3coKTtcbiAgICB9XG5cbiAgICBsZXQgdGV4U2hhcGUgPSB0ZXhEYXRhLnRleFNoYXBlO1xuICAgIGlmICh0ZXhTaGFwZSA9PSBudWxsKSB7XG4gICAgICAvLyBUaGlzIHRleFNoYXBlIG1heSBub3QgYmUgdGhlIGZpbmFsIHRleHR1cmUgc2hhcGUuIEZvciBwYWNrZWQgb3IgZGVuc2VcbiAgICAgIC8vIHRleHR1cmVzLCB0aGUgdGV4U2hhcGUgd2lsbCBiZSBjaGFuZ2VkIHdoZW4gdGV4dHVyZXMgYXJlIGNyZWF0ZWQuXG4gICAgICB0ZXhTaGFwZSA9IHdlYmdsX3V0aWwuZ2V0VGV4dHVyZVNoYXBlRnJvbUxvZ2ljYWxTaGFwZShzaGFwZSwgaXNQYWNrZWQpO1xuICAgICAgdGV4RGF0YS50ZXhTaGFwZSA9IHRleFNoYXBlO1xuICAgIH1cblxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgY29uc3Qgc2hhcGVBczNEID0gd2ViZ2xfdXRpbC5nZXRTaGFwZUFzM0Qoc2hhcGUpO1xuXG4gICAgICBsZXQgcHJvZ3JhbTtcbiAgICAgIGxldCB3aWR0aCA9IHRleFNoYXBlWzFdLCBoZWlnaHQgPSB0ZXhTaGFwZVswXTtcbiAgICAgIGNvbnN0IGlzQnl0ZUFycmF5ID1cbiAgICAgICAgICB2YWx1ZXMgaW5zdGFuY2VvZiBVaW50OEFycmF5IHx8IHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5O1xuXG4gICAgICAvLyB0ZXh0dXJlIGZvciBmbG9hdCBhcnJheSBpcyBQaHlzaWNhbFRleHR1cmVUeXBlLlBBQ0tFRF8yWDJfRkxPQVQzMiwgd2VcbiAgICAgIC8vIG5lZWQgdG8gbWFrZSBzdXJlIHRoZSB1cGxvYWQgdXNlcyB0aGUgc2FtZSBwYWNrZWQgc2l6ZVxuICAgICAgaWYgKGlzUGFja2VkIHx8ICFpc0J5dGVBcnJheSkge1xuICAgICAgICBbd2lkdGgsIGhlaWdodF0gPSB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICAgICAgICAgIHRleFNoYXBlWzBdLCB0ZXhTaGFwZVsxXSk7XG4gICAgICB9XG5cbiAgICAgIGlmIChpc1BhY2tlZCkge1xuICAgICAgICBwcm9ncmFtID0gbmV3IEVuY29kZU1hdHJpeFBhY2tlZFByb2dyYW0oc2hhcGVBczNELCBpc0J5dGVBcnJheSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBwcm9ncmFtID0gbmV3IEVuY29kZU1hdHJpeFByb2dyYW0oc2hhcGVBczNELCBpc0J5dGVBcnJheSk7XG4gICAgICB9XG5cbiAgICAgIC8vIFRleFNoYXBlIGZvciBmbG9hdCBhcnJheSBuZWVkcyB0byBiZSB0aGUgb3JpZ2luYWwgc2hhcGUsIHdoaWNoIGJ5dGVcbiAgICAgIC8vIGFycmF5IG5lZWRzIHRvIGJlIHBhY2tlZCBzaXplLiBUaGlzIGFsbG93IHRoZSBkYXRhIHVwbG9hZCBzaGFwZSB0byBiZVxuICAgICAgLy8gbWF0Y2hlZCB3aXRoIHRleHR1cmUgY3JlYXRpb24gbG9naWMuXG4gICAgICBjb25zdCB0ZW1wRGVuc2VJbnB1dFRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdID1cbiAgICAgICAgICBpc0J5dGVBcnJheSA/IFtoZWlnaHQsIHdpZHRoXSA6IHRleFNoYXBlO1xuICAgICAgY29uc3QgdGVtcERlbnNlSW5wdXRIYW5kbGUgPVxuICAgICAgICAgIHRoaXMubWFrZVRlbnNvckluZm8odGVtcERlbnNlSW5wdXRUZXhTaGFwZSwgZHR5cGUpO1xuICAgICAgY29uc3QgdGVtcERlbnNlSW5wdXRUZXhEYXRhID1cbiAgICAgICAgICB0aGlzLnRleERhdGEuZ2V0KHRlbXBEZW5zZUlucHV0SGFuZGxlLmRhdGFJZCk7XG4gICAgICBpZiAoaXNCeXRlQXJyYXkpIHtcbiAgICAgICAgdGVtcERlbnNlSW5wdXRUZXhEYXRhLnVzYWdlID0gVGV4dHVyZVVzYWdlLlBJWEVMUztcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRlbXBEZW5zZUlucHV0VGV4RGF0YS51c2FnZSA9IFRleHR1cmVVc2FnZS5VUExPQUQ7XG4gICAgICB9XG4gICAgICB0ZW1wRGVuc2VJbnB1dFRleERhdGEudGV4U2hhcGUgPSB0ZW1wRGVuc2VJbnB1dFRleFNoYXBlO1xuICAgICAgdGhpcy5ncGdwdS51cGxvYWREZW5zZU1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgICB0aGlzLmdldFRleHR1cmUodGVtcERlbnNlSW5wdXRIYW5kbGUuZGF0YUlkKSwgd2lkdGgsIGhlaWdodCxcbiAgICAgICAgICB2YWx1ZXMgYXMgVHlwZWRBcnJheSk7XG5cbiAgICAgIGNvbnN0IGN1c3RvbVZhbHVlcyA9IFtbaGVpZ2h0LCB3aWR0aF1dO1xuICAgICAgLy8gV2Ugd2FudCB0aGUgb3V0cHV0IHRvIHJlbWFpbiBwYWNrZWQgcmVnYXJkbGVzcyBvZiB0aGUgdmFsdWUgb2ZcbiAgICAgIC8vIFdFQkdMX1BBQ0suXG4gICAgICBjb25zdCBwcmV2ZW50RWFnZXJVbnBhY2tpbmcgPSB0cnVlO1xuICAgICAgY29uc3QgZW5jb2RlZE91dHB1dFRhcmdldCA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICAgIHByb2dyYW0sIFt0ZW1wRGVuc2VJbnB1dEhhbmRsZV0sIGR0eXBlLCBjdXN0b21WYWx1ZXMsXG4gICAgICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nKTtcblxuICAgICAgLy8gSGF2ZSB0aGUgb3JpZ2luYWwgdGV4dHVyZSBhc3N1bWUgdGhlIGlkZW50aXR5IG9mIHRoZSBlbmNvZGVkIG91dHB1dC5cbiAgICAgIGNvbnN0IG91dHB1dFRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGVuY29kZWRPdXRwdXRUYXJnZXQuZGF0YUlkKTtcbiAgICAgIHRleERhdGEudGV4U2hhcGUgPSBvdXRwdXRUZXhEYXRhLnRleFNoYXBlO1xuICAgICAgdGV4RGF0YS5pc1BhY2tlZCA9IG91dHB1dFRleERhdGEuaXNQYWNrZWQ7XG4gICAgICB0ZXhEYXRhLnVzYWdlID0gb3V0cHV0VGV4RGF0YS51c2FnZTtcblxuICAgICAgaWYgKCFlbnYoKS5nZXQoJ0VOR0lORV9DT01QSUxFX09OTFknKSkge1xuICAgICAgICB0ZXhEYXRhLnRleHR1cmUgPSBvdXRwdXRUZXhEYXRhLnRleHR1cmU7XG4gICAgICAgIC8vIE9uY2UgdXBsb2FkZWQsIGRvbid0IHN0b3JlIHRoZSB2YWx1ZXMgb24gY3B1LlxuICAgICAgICB0ZXhEYXRhLnZhbHVlcyA9IG51bGw7XG4gICAgICAgIHRoaXMudGV4RGF0YS5kZWxldGUoZW5jb2RlZE91dHB1dFRhcmdldC5kYXRhSWQpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5kaXNwb3NlRGF0YShlbmNvZGVkT3V0cHV0VGFyZ2V0LmRhdGFJZCk7XG4gICAgICB9XG5cbiAgICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odGVtcERlbnNlSW5wdXRIYW5kbGUpO1xuXG4gICAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgICAgdGhpcy51cGxvYWRXYWl0TXMgKz0gdXRpbC5ub3coKSAtIHN0YXJ0O1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBuZXdUZXh0dXJlID0gdGhpcy5hY3F1aXJlVGV4dHVyZSh0ZXhTaGFwZSwgdXNhZ2UsIGR0eXBlLCBpc1BhY2tlZCk7XG4gICAgICB0ZXhEYXRhLnRleHR1cmUgPSBuZXdUZXh0dXJlO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkOiBEYXRhSWQsIGZsb2F0MzJWYWx1ZXM/OiBGbG9hdDMyQXJyYXkpOlxuICAgICAgVHlwZWRBcnJheSB7XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7ZHR5cGV9ID0gdGV4RGF0YTtcblxuICAgIGlmIChmbG9hdDMyVmFsdWVzICE9IG51bGwpIHtcbiAgICAgIHRleERhdGEudmFsdWVzID0gZmxvYXQzMlRvVHlwZWRBcnJheShmbG9hdDMyVmFsdWVzLCBkdHlwZSBhcyAnZmxvYXQzMicpO1xuICAgIH1cbiAgICByZXR1cm4gdGV4RGF0YS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbiAgfVxuXG4gIHByaXZhdGUgYWNxdWlyZVRleHR1cmUoXG4gICAgICB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSwgdGV4VHlwZTogVGV4dHVyZVVzYWdlLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICBpc1BhY2tlZDogYm9vbGVhbik6IFRleHR1cmUge1xuICAgIHRoaXMubnVtQnl0ZXNJbkdQVSArPSB0aGlzLmNvbXB1dGVCeXRlcyh0ZXhTaGFwZSwgZHR5cGUpO1xuICAgIGlmICghdGhpcy53YXJuZWRBYm91dE1lbW9yeSAmJlxuICAgICAgICB0aGlzLm51bUJ5dGVzSW5HUFUgPiB0aGlzLm51bU1CQmVmb3JlV2FybmluZyAqIDEwMjQgKiAxMDI0KSB7XG4gICAgICBjb25zdCBtYiA9ICh0aGlzLm51bUJ5dGVzSW5HUFUgLyAxMDI0IC8gMTAyNCkudG9GaXhlZCgyKTtcbiAgICAgIHRoaXMud2FybmVkQWJvdXRNZW1vcnkgPSB0cnVlO1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBIaWdoIG1lbW9yeSB1c2FnZSBpbiBHUFU6ICR7bWJ9IE1CLCBgICtcbiAgICAgICAgICBgbW9zdCBsaWtlbHkgZHVlIHRvIGEgbWVtb3J5IGxlYWtgKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudGV4dHVyZU1hbmFnZXIuYWNxdWlyZVRleHR1cmUodGV4U2hhcGUsIHRleFR5cGUsIGlzUGFja2VkKTtcbiAgfVxuXG4gIHByaXZhdGUgY29tcHV0ZUJ5dGVzKHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCBkdHlwZTogRGF0YVR5cGUpIHtcbiAgICByZXR1cm4gc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHV0aWwuYnl0ZXNQZXJFbGVtZW50KGR0eXBlKTtcbiAgfVxuXG4gIGNoZWNrQ29tcGlsZUNvbXBsZXRpb24oKSB7XG4gICAgZm9yIChjb25zdCBbLCBiaW5hcnldIG9mIE9iamVjdC5lbnRyaWVzKHRoaXMuYmluYXJ5Q2FjaGUpKSB7XG4gICAgICB0aGlzLmNoZWNrQ29tcGxldGlvbl8oYmluYXJ5KTtcbiAgICB9XG4gIH1cblxuICBhc3luYyBjaGVja0NvbXBpbGVDb21wbGV0aW9uQXN5bmMoKTogUHJvbWlzZTxib29sZWFuW10+IHtcbiAgICBjb25zdCBwcyA9IFtdO1xuICAgIGlmICh0aGlzLmdwZ3B1LnBhcmFsbGVsQ29tcGlsYXRpb25FeHRlbnNpb24pIHtcbiAgICAgIGZvciAoY29uc3QgWywgYmluYXJ5XSBvZiBPYmplY3QuZW50cmllcyh0aGlzLmJpbmFyeUNhY2hlKSkge1xuICAgICAgICBwcy5wdXNoKHRoaXMuY2hlY2tDb21wbGV0aW9uQXN5bmNfKGJpbmFyeSkpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIFByb21pc2UuYWxsKHBzKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZm9yIChjb25zdCBbLCBiaW5hcnldIG9mIE9iamVjdC5lbnRyaWVzKHRoaXMuYmluYXJ5Q2FjaGUpKSB7XG4gICAgICAgIGNvbnN0IHA6IFByb21pc2U8Ym9vbGVhbj4gPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4ge1xuICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICB0aGlzLmNoZWNrQ29tcGxldGlvbl8oYmluYXJ5KTtcbiAgICAgICAgICAgIHJlc29sdmUodHJ1ZSk7XG4gICAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICAgIHBzLnB1c2gocCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gUHJvbWlzZS5hbGwocHMpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgY2hlY2tDb21wbGV0aW9uQXN5bmNfKGJpbmFyeTogR1BHUFVCaW5hcnkpOiBQcm9taXNlPGJvb2xlYW4+IHtcbiAgICBpZiAodGhpcy5ncGdwdS5nbC5nZXRQcm9ncmFtUGFyYW1ldGVyKFxuICAgICAgICAgICAgYmluYXJ5LndlYkdMUHJvZ3JhbSxcbiAgICAgICAgICAgIHRoaXMuZ3BncHUucGFyYWxsZWxDb21waWxhdGlvbkV4dGVuc2lvbi5DT01QTEVUSU9OX1NUQVRVU19LSFIpKSB7XG4gICAgICByZXR1cm4gdGhpcy5jaGVja0NvbXBsZXRpb25fKGJpbmFyeSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGF3YWl0IG5leHRGcmFtZSgpO1xuICAgICAgcmV0dXJuIHRoaXMuY2hlY2tDb21wbGV0aW9uQXN5bmNfKGJpbmFyeSk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBjaGVja0NvbXBsZXRpb25fKGJpbmFyeTogR1BHUFVCaW5hcnkpOiBib29sZWFuIHtcbiAgICBpZiAodGhpcy5ncGdwdS5nbC5nZXRQcm9ncmFtUGFyYW1ldGVyKFxuICAgICAgICAgICAgYmluYXJ5LndlYkdMUHJvZ3JhbSwgdGhpcy5ncGdwdS5nbC5MSU5LX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgICBjb25zb2xlLmxvZyh0aGlzLmdwZ3B1LmdsLmdldFByb2dyYW1JbmZvTG9nKGJpbmFyeS53ZWJHTFByb2dyYW0pKTtcbiAgICAgIGlmICh0aGlzLmdwZ3B1LmdsLmdldFNoYWRlclBhcmFtZXRlcihcbiAgICAgICAgICAgICAgYmluYXJ5LmZyYWdtZW50U2hhZGVyLCB0aGlzLmdwZ3B1LmdsLkNPTVBJTEVfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICAgICAgd2ViZ2xfdXRpbC5sb2dTaGFkZXJTb3VyY2VBbmRJbmZvTG9nKFxuICAgICAgICAgICAgYmluYXJ5LnNvdXJjZSxcbiAgICAgICAgICAgIHRoaXMuZ3BncHUuZ2wuZ2V0U2hhZGVySW5mb0xvZyhiaW5hcnkuZnJhZ21lbnRTaGFkZXIpKTtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSBmcmFnbWVudCBzaGFkZXIuJyk7XG4gICAgICB9XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBsaW5rIHZlcnRleCBhbmQgZnJhZ21lbnQgc2hhZGVycy4nKTtcbiAgICB9XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICBnZXRVbmlmb3JtTG9jYXRpb25zKCkge1xuICAgIGZvciAoY29uc3QgWywgYmluYXJ5XSBvZiBPYmplY3QuZW50cmllcyh0aGlzLmJpbmFyeUNhY2hlKSkge1xuICAgICAgY29uc3Qge1xuICAgICAgICB1bmlmb3JtTG9jYXRpb25zLFxuICAgICAgICBjdXN0b21Vbmlmb3JtTG9jYXRpb25zLFxuICAgICAgICBpbmZMb2MsXG4gICAgICAgIG5hbkxvYyxcbiAgICAgICAgaW5TaGFwZXNMb2NhdGlvbnMsXG4gICAgICAgIGluVGV4U2hhcGVzTG9jYXRpb25zLFxuICAgICAgICBvdXRTaGFwZUxvY2F0aW9uLFxuICAgICAgICBvdXRTaGFwZVN0cmlkZXNMb2NhdGlvbixcbiAgICAgICAgb3V0VGV4U2hhcGVMb2NhdGlvblxuICAgICAgfSA9IGdldFVuaWZvcm1Mb2NhdGlvbnModGhpcy5ncGdwdSwgYmluYXJ5LnByb2dyYW0sIGJpbmFyeS53ZWJHTFByb2dyYW0pO1xuICAgICAgYmluYXJ5LnVuaWZvcm1Mb2NhdGlvbnMgPSB1bmlmb3JtTG9jYXRpb25zO1xuICAgICAgYmluYXJ5LmN1c3RvbVVuaWZvcm1Mb2NhdGlvbnMgPSBjdXN0b21Vbmlmb3JtTG9jYXRpb25zO1xuICAgICAgYmluYXJ5LmluZkxvYyA9IGluZkxvYztcbiAgICAgIGJpbmFyeS5uYW5Mb2MgPSBuYW5Mb2M7XG4gICAgICBiaW5hcnkuaW5TaGFwZXNMb2NhdGlvbnMgPSBpblNoYXBlc0xvY2F0aW9ucztcbiAgICAgIGJpbmFyeS5pblRleFNoYXBlc0xvY2F0aW9ucyA9IGluVGV4U2hhcGVzTG9jYXRpb25zO1xuICAgICAgYmluYXJ5Lm91dFNoYXBlTG9jYXRpb24gPSBvdXRTaGFwZUxvY2F0aW9uO1xuICAgICAgYmluYXJ5Lm91dFNoYXBlU3RyaWRlc0xvY2F0aW9uID0gb3V0U2hhcGVTdHJpZGVzTG9jYXRpb247XG4gICAgICBiaW5hcnkub3V0VGV4U2hhcGVMb2NhdGlvbiA9IG91dFRleFNoYXBlTG9jYXRpb247XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIFRGLmpzIHRlbnNvciBvdXQgb2YgYW4gZXhpc3RpbmcgV2ViR0wgdGV4dHVyZS4gQSBuZXcgdGV4dHVyZSB3aWxsXG4gICAqIGJlIGNyZWF0ZWQuXG4gICAqL1xuICBvdmVycmlkZSBjcmVhdGVUZW5zb3JGcm9tR1BVRGF0YShcbiAgICAgIHZhbHVlczogV2ViR0xEYXRhLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6IFRlbnNvciB7XG4gICAgdmFsdWVzLmNoYW5uZWxzID0gdmFsdWVzLmNoYW5uZWxzIHx8ICdSR0JBJztcbiAgICBjb25zdCB7dGV4dHVyZSwgaGVpZ2h0LCB3aWR0aCwgY2hhbm5lbHN9ID0gdmFsdWVzO1xuICAgIGNvbnN0IGJhY2tlbmQgPSBlbmdpbmUoKS5iYWNrZW5kIGFzIE1hdGhCYWNrZW5kV2ViR0w7XG5cbiAgICAvLyBIYXZlIHRvIHRocm93IGFuIGVycm9yLCBvdGhlcndpc2UgV2ViR0wganVzdCB3YXJucyBhbmQgcmV0dXJucyB3cm9uZ1xuICAgIC8vIHZhbHVlcy5cbiAgICBpZiAoIWJhY2tlbmQuZ3BncHUuZ2wuaXNUZXh0dXJlKHRleHR1cmUpKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYFRoZSB0ZXh0dXJlIGlzIGludmFsaWQuIEFsc28sIHBsZWFzZSBtYWtlIHN1cmUgdGhlIHRleHR1cmUgYW5kIGAgK1xuICAgICAgICAgIGB0aGUgVEZKUyBXZWJHTCBiYWNrZW5kIGFyZSB1c2luZyB0aGUgc2FtZSBjYW52YXMuIElmIHlvdSB3YW50IHRvIGAgK1xuICAgICAgICAgIGB1c2UgeW91ciBvd24gY3VzdG9tIGNhbnZhcywgeW91IGhhdmUgdG8gY3JlYXRlIGFuZCB1c2UgdGhlIGN1c3RvbSBgICtcbiAgICAgICAgICBgVEZKUyBXZWJHTCBiYWNrZW5kIGNyZWF0ZWQgZnJvbSB0aGUgY2FudmFzIHRocm91Z2ggYCArXG4gICAgICAgICAgYCduZXcgdGYuTWF0aEJhY2tlbmRXZWJHTChjdXN0b21DYW52YXMpJy5gKTtcbiAgICB9XG5cbiAgICBjb25zdCBkYXRhSWQgPVxuICAgICAgICBiYWNrZW5kLndyaXRlVGV4dHVyZSh0ZXh0dXJlLCBzaGFwZSwgZHR5cGUsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzKTtcbiAgICByZXR1cm4gZW5naW5lKCkubWFrZVRlbnNvckZyb21EYXRhSWQoZGF0YUlkLCBzaGFwZSwgZHR5cGUsIGJhY2tlbmQpO1xuICB9XG59XG5cbmZ1bmN0aW9uIGZsb2F0MzJUb1R5cGVkQXJyYXk8RCBleHRlbmRzIE51bWVyaWNEYXRhVHlwZT4oXG4gICAgYTogRmxvYXQzMkFycmF5LCBkdHlwZTogRCk6IHRmLkRhdGFUeXBlTWFwW0RdIHtcbiAgaWYgKGR0eXBlID09PSAnZmxvYXQzMicgfHwgZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGEgYXMgdGYuRGF0YVR5cGVNYXBbRF07XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicgfHwgZHR5cGUgPT09ICdib29sJykge1xuICAgIGNvbnN0IHJlc3VsdCA9IChkdHlwZSA9PT0gJ2ludDMyJykgPyBuZXcgSW50MzJBcnJheShhLmxlbmd0aCkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXcgVWludDhBcnJheShhLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXN1bHQubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFtpXSA9IE1hdGgucm91bmQoYVtpXSk7XG4gICAgfVxuICAgIHJldHVybiByZXN1bHQgYXMgdGYuRGF0YVR5cGVNYXBbRF07XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGR0eXBlICR7ZHR5cGV9YCk7XG4gIH1cbn1cbiJdfQ==