/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { complex } from '../ops/complex';
import { tensor } from '../ops/tensor';
import { sizeFromShape } from '../util';
import { DTYPE_VALUE_SIZE_MAP } from './types';
/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;
/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(tensors, group) {
    // TODO(adarob, cais): Support quantization.
    const specs = [];
    const dataPromises = [];
    const names = Array.isArray(tensors) ?
        tensors.map(tensor => tensor.name) :
        Object.keys(tensors);
    for (let i = 0; i < names.length; ++i) {
        const name = names[i];
        const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
        if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
            t.dtype !== 'string' && t.dtype !== 'complex64') {
            throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
        }
        const spec = { name, shape: t.shape, dtype: t.dtype };
        if (t.dtype === 'string') {
            const utf8bytes = new Promise(async (resolve) => {
                const vals = await t.bytes();
                const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
                    NUM_BYTES_STRING_LENGTH * vals.length;
                const bytes = new Uint8Array(totalNumBytes);
                let offset = 0;
                for (let i = 0; i < vals.length; i++) {
                    const val = vals[i];
                    const bytesOfLength = new Uint8Array(new Uint32Array([val.length]).buffer);
                    bytes.set(bytesOfLength, offset);
                    offset += NUM_BYTES_STRING_LENGTH;
                    bytes.set(val, offset);
                    offset += val.length;
                }
                resolve(bytes);
            });
            dataPromises.push(utf8bytes);
        }
        else {
            dataPromises.push(t.data());
        }
        if (group != null) {
            spec.group = group;
        }
        specs.push(spec);
    }
    const tensorValues = await Promise.all(dataPromises);
    return { data: concatenateTypedArrays(tensorValues), specs };
}
/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(buffer, specs) {
    // TODO(adarob, cais): Support quantization.
    const out = {};
    let float16Decode;
    let offset = 0;
    for (const spec of specs) {
        const name = spec.name;
        const dtype = spec.dtype;
        const shape = spec.shape;
        const size = sizeFromShape(shape);
        let values;
        if ('quantization' in spec) {
            const quantization = spec.quantization;
            if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                if (!('min' in quantization && 'scale' in quantization)) {
                    throw new Error(`Weight ${spec.name} with quantization ${quantization.dtype} ` +
                        `doesn't have corresponding metadata min and scale.`);
                }
            }
            else if (quantization.dtype === 'float16') {
                if (dtype !== 'float32') {
                    throw new Error(`Weight ${spec.name} is quantized with ${quantization.dtype} ` +
                        `which only supports weights of type float32 not ${dtype}.`);
                }
            }
            else {
                throw new Error(`Weight ${spec.name} has unknown ` +
                    `quantization dtype ${quantization.dtype}. ` +
                    `Supported quantization dtypes are: ` +
                    `'uint8', 'uint16', and 'float16'.`);
            }
            const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
            const byteBuffer = buffer.slice(offset, offset + size * quantizationSizeFactor);
            const quantizedArray = (quantization.dtype === 'uint8') ?
                new Uint8Array(byteBuffer) :
                new Uint16Array(byteBuffer);
            if (dtype === 'float32') {
                if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                    values = new Float32Array(quantizedArray.length);
                    for (let i = 0; i < quantizedArray.length; i++) {
                        const v = quantizedArray[i];
                        values[i] = v * quantization.scale + quantization.min;
                    }
                }
                else if (quantization.dtype === 'float16') {
                    if (float16Decode === undefined) {
                        float16Decode = getFloat16Decoder();
                    }
                    values = float16Decode(quantizedArray);
                }
                else {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type float32.`);
                }
            }
            else if (dtype === 'int32') {
                if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type int32.`);
                }
                values = new Int32Array(quantizedArray.length);
                for (let i = 0; i < quantizedArray.length; i++) {
                    const v = quantizedArray[i];
                    values[i] = Math.round(v * quantization.scale + quantization.min);
                }
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * quantizationSizeFactor;
        }
        else if (dtype === 'string') {
            const size = sizeFromShape(spec.shape);
            values = [];
            for (let i = 0; i < size; i++) {
                const byteLength = new Uint32Array(buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
                offset += NUM_BYTES_STRING_LENGTH;
                const bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
                values.push(bytes);
                offset += byteLength;
            }
        }
        else {
            const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
            const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
            if (dtype === 'float32') {
                values = new Float32Array(byteBuffer);
            }
            else if (dtype === 'int32') {
                values = new Int32Array(byteBuffer);
            }
            else if (dtype === 'bool') {
                values = new Uint8Array(byteBuffer);
            }
            else if (dtype === 'complex64') {
                values = new Float32Array(byteBuffer);
                const real = new Float32Array(values.length / 2);
                const image = new Float32Array(values.length / 2);
                for (let i = 0; i < real.length; i++) {
                    real[i] = values[i * 2];
                    image[i] = values[i * 2 + 1];
                }
                const realTensor = tensor(real, shape, 'float32');
                const imageTensor = tensor(image, shape, 'float32');
                out[name] = complex(realTensor, imageTensor);
                realTensor.dispose();
                imageTensor.dispose();
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * dtypeFactor;
        }
        if (dtype !== 'complex64') {
            out[name] = tensor(values, shape, dtype);
        }
    }
    return out;
}
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
export function concatenateTypedArrays(xs) {
    // TODO(adarob, cais): Support quantization.
    if (xs === null) {
        throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
    }
    let totalByteLength = 0;
    // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
    // can have a different byte length from that of the `TypedArray` itself,
    // for example, when the `TypedArray` is created from an offset in an
    // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
    // the `TypedArray` in byte length. If an element of `xs` does not show
    // this property, a new `TypedArray` that satisfy this property will be
    // constructed and pushed into `normalizedXs`.
    const normalizedXs = [];
    xs.forEach((x) => {
        totalByteLength += x.byteLength;
        // tslint:disable:no-any
        normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
            new x.constructor(x));
        if (!(x instanceof Float32Array || x instanceof Int32Array ||
            x instanceof Uint8Array)) {
            throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
        }
        // tslint:enable:no-any
    });
    const y = new Uint8Array(totalByteLength);
    let offset = 0;
    normalizedXs.forEach((x) => {
        y.set(new Uint8Array(x.buffer), offset);
        offset += x.byteLength;
    });
    return y.buffer;
}
// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
        typeof btoa === 'undefined');
/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
export function stringByteLength(str) {
    if (useNodeBuffer) {
        return Buffer.byteLength(str);
    }
    return new Blob([str]).size;
}
/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer) {
    if (useNodeBuffer) {
        return Buffer.from(buffer).toString('base64');
    }
    const buf = new Uint8Array(buffer);
    let s = '';
    for (let i = 0, l = buf.length; i < l; i++) {
        s += String.fromCharCode(buf[i]);
    }
    return btoa(s);
}
/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str) {
    if (useNodeBuffer) {
        const buf = Buffer.from(str, 'base64');
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const s = atob(str);
    const buffer = new Uint8Array(s.length);
    for (let i = 0; i < s.length; ++i) {
        buffer.set([s.charCodeAt(i)], i);
    }
    return buffer.buffer;
}
/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
export function concatenateArrayBuffers(buffers) {
    if (buffers.length === 1) {
        return buffers[0];
    }
    let totalByteLength = 0;
    buffers.forEach((buffer) => {
        totalByteLength += buffer.byteLength;
    });
    const temp = new Uint8Array(totalByteLength);
    let offset = 0;
    buffers.forEach((buffer) => {
        temp.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    });
    return temp.buffer;
}
/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
export function basename(path) {
    const SEPARATOR = '/';
    path = path.trim();
    while (path.endsWith(SEPARATOR)) {
        path = path.slice(0, path.length - 1);
    }
    const items = path.split(SEPARATOR);
    return items[items.length - 1];
}
/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
export function getModelJSONForModelArtifacts(artifacts, manifest) {
    const result = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: manifest
    };
    if (artifacts.signature != null) {
        result.signature = artifacts.signature;
    }
    if (artifacts.userDefinedMetadata != null) {
        result.userDefinedMetadata = artifacts.userDefinedMetadata;
    }
    if (artifacts.modelInitializer != null) {
        result.modelInitializer = artifacts.modelInitializer;
    }
    if (artifacts.initializerSignature != null) {
        result.initializerSignature = artifacts.initializerSignature;
    }
    if (artifacts.trainingConfig != null) {
        result.trainingConfig = artifacts.trainingConfig;
    }
    return result;
}
/**
 * Create `ModelArtifacts` from a JSON file and weights.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param weightSpecs The list of WeightsManifestEntry for the model. Must be
 *     passed if the modelJSON has a weightsManifest.
 * @param weightData An ArrayBuffer of weight data for the model corresponding
 *     to the weights in weightSpecs. Must be passed if the modelJSON has a
 *     weightsManifest.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export function getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData) {
    const modelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
    };
    if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
    }
    if (modelJSON.weightsManifest != null) {
        if (!weightSpecs) {
            throw new Error('modelJSON has weightsManifest but weightSpecs is null');
        }
        if (!weightData) {
            throw new Error('modelJSON has weightsManifest but weightData is null');
        }
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
    }
    if (modelJSON.signature != null) {
        modelArtifacts.signature = modelJSON.signature;
    }
    if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
    }
    if (modelJSON.modelInitializer != null) {
        modelArtifacts.modelInitializer = modelJSON.modelInitializer;
    }
    if (modelJSON.initializerSignature != null) {
        modelArtifacts.initializerSignature = modelJSON.initializerSignature;
    }
    return modelArtifacts;
}
/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export async function getModelArtifactsForJSON(modelJSON, loadWeights) {
    let weightSpecs;
    let weightData;
    if (modelJSON.weightsManifest != null) {
        [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest);
    }
    return getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData);
}
/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts) {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
        throw new Error('Expected JSON model topology, received ArrayBuffer.');
    }
    return {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelArtifacts.modelTopology == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData == null ?
            0 :
            modelArtifacts.weightData.byteLength,
    };
}
/**
 * Concatenate the weights stored in a WeightsManifestConfig into a list of
 * WeightsManifestEntry
 *
 * @param weightsManifest The WeightsManifestConfig to extract weights from.
 * @returns A list of WeightsManifestEntry of the weights in the weightsManifest
 */
export function getWeightSpecs(weightsManifest) {
    const weightSpecs = [];
    for (const entry of weightsManifest) {
        weightSpecs.push(...entry.weights);
    }
    return weightSpecs;
}
/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable() {
    const convertMantissa = (i) => {
        let m = i << 13;
        let e = 0;
        while ((m & 0x00800000) === 0) {
            e -= 0x00800000;
            m <<= 1;
        }
        m &= ~0x00800000;
        e += 0x38800000;
        return m | e;
    };
    const mantisaTable = new Uint32Array(2048);
    mantisaTable[0] = 0;
    for (let i = 1; i < 1024; i++) {
        mantisaTable[i] = convertMantissa(i);
    }
    for (let i = 1024; i < 2048; i++) {
        mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
    }
    return mantisaTable;
}
/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable() {
    const exponentTable = new Uint32Array(64);
    exponentTable[0] = 0;
    exponentTable[31] = 0x47800000;
    exponentTable[32] = 0x80000000;
    exponentTable[63] = 0xc7800000;
    for (let i = 1; i < 31; i++) {
        exponentTable[i] = i << 23;
    }
    for (let i = 33; i < 63; i++) {
        exponentTable[i] = 0x80000000 + ((i - 32) << 23);
    }
    return exponentTable;
}
/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable() {
    const offsetTable = new Uint32Array(64);
    for (let i = 0; i < 64; i++) {
        offsetTable[i] = 1024;
    }
    offsetTable[0] = offsetTable[32] = 0;
    return offsetTable;
}
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
export function getFloat16Decoder() {
    // Algorithm is based off of
    // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
    // Cache lookup tables
    const mantisaTable = computeFloat16MantisaTable();
    const exponentTable = computeFloat16ExponentTable();
    const offsetTable = computeFloat16OffsetTable();
    return (quantizedArray) => {
        const buffer = new ArrayBuffer(4 * quantizedArray.length);
        const bufferUint32View = new Uint32Array(buffer);
        for (let index = 0; index < quantizedArray.length; index++) {
            const float16Bits = quantizedArray[index];
            const float32Bits = mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
                exponentTable[float16Bits >> 10];
            bufferUint32View[index] = float32Bits;
        }
        return new Float32Array(buffer);
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW9fdXRpbHMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2lvL2lvX3V0aWxzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUN2QyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBR3JDLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdEMsT0FBTyxFQUFDLG9CQUFvQixFQUEwRyxNQUFNLFNBQVMsQ0FBQztBQUV0Siw4RUFBOEU7QUFDOUUsTUFBTSx1QkFBdUIsR0FBRyxDQUFDLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxNQUFNLENBQUMsS0FBSyxVQUFVLGFBQWEsQ0FDL0IsT0FBcUMsRUFBRSxLQUFtQjtJQUU1RCw0Q0FBNEM7SUFDNUMsTUFBTSxLQUFLLEdBQTJCLEVBQUUsQ0FBQztJQUN6QyxNQUFNLFlBQVksR0FBK0IsRUFBRSxDQUFDO0lBRXBELE1BQU0sS0FBSyxHQUFhLEtBQUssQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUM1QyxPQUFPLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUV6QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNyQyxNQUFNLElBQUksR0FBRyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsTUFBTSxDQUFDLEdBQUcsS0FBSyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMs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QVcsR0FBRyxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDMUMsTUFBTSxXQUFXLEdBQ2IsWUFBWSxDQUFDLFdBQVcsQ0FBQyxXQUFXLElBQUksRUFBRSxDQUFDLEdBQUcsQ0FBQyxXQUFXLEdBQUcsS0FBSyxDQUFDLENBQUM7Z0JBQ3BFLGFBQWEsQ0FBQyxXQUFXLElBQUksRUFBRSxDQUFDLENBQUM7WUFDckMsZ0JBQWdCLENBQUMsS0FBSyxDQUFDLEdBQUcsV0FBVyxDQUFDO1NBQ3ZDO1FBQ0QsT0FBTyxJQUFJLFlBQVksQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUNsQyxDQUFDLENBQUM7QUFDSixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2NvbXBsZXh9IGZyb20gJy4uL29wcy9jb21wbGV4JztcbmltcG9ydCB7dGVuc29yfSBmcm9tICcuLi9vcHMvdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3IsIE5hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtUeXBlZEFycmF5fSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge3NpemVGcm9tU2hhcGV9IGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge0RUWVBFX1ZBTFVFX1NJWkVfTUFQLCBNb2RlbEFydGlmYWN0cywgTW9kZWxBcnRpZmFjdHNJbmZvLCBNb2RlbEpTT04sIFdlaWdodEdyb3VwLCBXZWlnaHRzTWFuaWZlc3RDb25maWcsIFdlaWdodHNNYW5pZmVzdEVudHJ5fSBmcm9tICcuL3R5cGVzJztcblxuLyoqIE51bWJlciBvZiBieXRlcyByZXNlcnZlZCBmb3IgdGhlIGxlbmd0aCBvZiB0aGUgc3RyaW5nLiAoMzJiaXQgaW50ZWdlcikuICovXG5jb25zdCBOVU1fQllURVNfU1RSSU5HX0xFTkdUSCA9IDQ7XG5cbi8qKlxuICogRW5jb2RlIGEgbWFwIGZyb20gbmFtZXMgdG8gd2VpZ2h0IHZhbHVlcyBhcyBhbiBBcnJheUJ1ZmZlciwgYWxvbmcgd2l0aCBhblxuICogYEFycmF5YCBvZiBgV2VpZ2h0c01hbmlmZXN0RW50cnlgIGFzIHNwZWNpZmljYXRpb24gb2YgdGhlIGVuY29kZWQgd2VpZ2h0cy5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGRvZXMgbm90IHBlcmZvcm0gc2hhcmRpbmcuXG4gKlxuICogVGhpcyBmdW5jdGlvbiBpcyB0aGUgcmV2ZXJzZSBvZiBgZGVjb2RlV2VpZ2h0c2AuXG4gKlxuICogQHBhcmFtIHRlbnNvcnMgQSBtYXAgKFwiZGljdFwiKSBmcm9tIG5hbWVzIHRvIHRlbnNvcnMuXG4gKiBAcGFyYW0gZ3JvdXAgR3JvdXAgdG8gd2hpY2ggdGhlIHdlaWdodHMgYmVsb25nIChvcHRpb25hbCkuXG4gKiBAcmV0dXJucyBBIGBQcm9taXNlYCBvZlxuICogICAtIEEgZmxhdCBgQXJyYXlCdWZmZXJgIHdpdGggYWxsIHRoZSBiaW5hcnkgdmFsdWVzIG9mIHRoZSBgVGVuc29yYHNcbiAqICAgICBjb25jYXRlbmF0ZWQuXG4gKiAgIC0gQW4gYEFycmF5YCBvZiBgV2VpZ2h0TWFuaWZlc3RFbnRyeWBzLCBjYXJyeWluZyBpbmZvcm1hdGlvbiBpbmNsdWRpbmdcbiAqICAgICB0ZW5zb3IgbmFtZXMsIGBkdHlwZWBzIGFuZCBzaGFwZXMuXG4gKiBAdGhyb3dzIEVycm9yOiBvbiB1bnN1cHBvcnRlZCB0ZW5zb3IgYGR0eXBlYC5cbiAqL1xuZXhwb3J0IGFzeW5jIGZ1bmN0aW9uIGVuY29kZVdlaWdodHMoXG4gICAgdGVuc29yczogTmFtZWRUZW5zb3JNYXB8TmFtZWRUZW5zb3JbXSwgZ3JvdXA/OiBXZWlnaHRHcm91cCk6XG4gICAgUHJvbWlzZTx7ZGF0YTogQXJyYXlCdWZmZXIsIHNwZWNzOiBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdfT4ge1xuICAvLyBUT0RPKGFkYXJvYiwgY2Fpcyk6IFN1cHBvcnQgcXVhbnRpemF0aW9uLlxuICBjb25zdCBzcGVjczogV2VpZ2h0c01hbmlmZXN0RW50cnlbXSA9IFtdO1xuICBjb25zdCBkYXRhUHJvbWlzZXM6IEFycmF5PFByb21pc2U8VHlwZWRBcnJheT4+ID0gW107XG5cbiAgY29uc3QgbmFtZXM6IHN0cmluZ1tdID0gQXJyYXkuaXNBcnJheSh0ZW5zb3JzKSA/XG4gICAgICB0ZW5zb3JzLm1hcCh0ZW5zb3IgPT4gdGVuc29yLm5hbWUpIDpcbiAgICAgIE9iamVjdC5rZXlzKHRlbnNvcnMpO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbmFtZXMubGVuZ3RoOyArK2kpIHtcbiAgICBjb25zdCBuYW1lID0gbmFtZXNbaV07XG4gICAgY29uc3QgdCA9IEFycmF5LmlzQXJyYXkodGVuc29ycykgPyB0ZW5zb3JzW2ldLnRlbnNvciA6IHRlbnNvcnNbbmFtZV07XG4gICAgaWYgKHQuZHR5cGUgIT09ICdmbG9hdDMyJyAmJiB0LmR0eXBlICE9PSAnaW50MzInICYmIHQuZHR5cGUgIT09ICdib29sJyAmJlxuICAgICAgICB0LmR0eXBlICE9PSAnc3RyaW5nJyAmJiB0LmR0eXBlICE9PSAnY29tcGxleDY0Jykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBkdHlwZSBpbiB3ZWlnaHQgJyR7bmFtZX0nOiAke3QuZHR5cGV9YCk7XG4gICAgfVxuICAgIGNvbnN0IHNwZWM6IFdlaWdodHNNYW5pZmVzdEVudHJ5ID0ge25hbWUsIHNoYXBlOiB0LnNoYXBlLCBkdHlwZTogdC5kdHlwZX07XG4gICAgaWYgKHQuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICBjb25zdCB1dGY4Ynl0ZXMgPSBuZXcgUHJvbWlzZTxUeXBlZEFycmF5Pihhc3luYyByZXNvbHZlID0+IHtcbiAgICAgICAgY29uc3QgdmFscyA9IGF3YWl0IHQuYnl0ZXMoKSBhcyBVaW50OEFycmF5W107XG4gICAgICAgIGNvbnN0IHRvdGFsTnVtQnl0ZXMgPSB2YWxzLnJlZHVjZSgocCwgYykgPT4gcCArIGMubGVuZ3RoLCAwKSArXG4gICAgICAgICAgICBOVU1fQllURVNfU1RSSU5HX0xFTkdUSCAqIHZhbHMubGVuZ3RoO1xuICAgICAgICBjb25zdCBieXRlcyA9IG5ldyBVaW50OEFycmF5KHRvdGFsTnVtQnl0ZXMpO1xuICAgICAgICBsZXQgb2Zmc2V0ID0gMDtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWxzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgY29uc3QgdmFsID0gdmFsc1tpXTtcbiAgICAgICAgICBjb25zdCBieXRlc09mTGVuZ3RoID1cbiAgICAgICAgICAgICAgbmV3IFVpbnQ4QXJyYXkobmV3IFVpbnQzMkFycmF5KFt2YWwubGVuZ3RoXSkuYnVmZmVyKTtcbiAgICAgICAgICBieXRlcy5zZXQoYnl0ZXNPZkxlbmd0aCwgb2Zmc2V0KTtcbiAgICAgICAgICBvZmZzZXQgKz0gTlVNX0JZVEVTX1NUUklOR19MRU5HVEg7XG4gICAgICAgICAgYnl0ZXMuc2V0KHZhbCwgb2Zmc2V0KTtcbiAgICAgICAgICBvZmZzZXQgKz0gdmFsLmxlbmd0aDtcbiAgICAgICAgfVxuICAgICAgICByZXNvbHZlKGJ5dGVzKTtcbiAgICAgIH0pO1xuICAgICAgZGF0YVByb21pc2VzLnB1c2godXRmOGJ5dGVzKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZGF0YVByb21pc2VzLnB1c2godC5kYXRhKCkpO1xuICAgIH1cbiAgICBpZiAoZ3JvdXAgIT0gbnVsbCkge1xuICAgICAgc3BlYy5ncm91cCA9IGdyb3VwO1xuICAgIH1cbiAgICBzcGVjcy5wdXNoKHNwZWMpO1xuICB9XG5cbiAgY29uc3QgdGVuc29yVmFsdWVzID0gYXdhaXQgUHJvbWlzZS5hbGwoZGF0YVByb21pc2VzKTtcbiAgcmV0dXJuIHtkYXRhOiBjb25jYXRlbmF0ZVR5cGVkQXJyYXlzKHRlbnNvclZhbHVlcyksIHNwZWNzfTtcbn1cblxuLyoqXG4gKiBEZWNvZGUgZmxhdCBBcnJheUJ1ZmZlciBhcyB3ZWlnaHRzLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gZG9lcyBub3QgaGFuZGxlIHNoYXJkaW5nLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gaXMgdGhlIHJldmVyc2Ugb2YgYGVuY29kZVdlaWdodHNgLlxuICpcbiAqIEBwYXJhbSBidWZmZXIgQSBmbGF0IEFycmF5QnVmZmVyIGNhcnJ5aW5nIHRoZSBiaW5hcnkgdmFsdWVzIG9mIHRoZSB0ZW5zb3JzXG4gKiAgIGNvbmNhdGVuYXRlZCBpbiB0aGUgb3JkZXIgc3BlY2lmaWVkIGluIGBzcGVjc2AuXG4gKiBAcGFyYW0gc3BlY3MgU3BlY2lmaWNhdGlvbnMgb2YgdGhlIG5hbWVzLCBkdHlwZXMgYW5kIHNoYXBlcyBvZiB0aGUgdGVuc29yc1xuICogICB3aG9zZSB2YWx1ZSBhcmUgZW5jb2RlZCBieSBgYnVmZmVyYC5cbiAqIEByZXR1cm4gQSBtYXAgZnJvbSB0ZW5zb3IgbmFtZSB0byB0ZW5zb3IgdmFsdWUsIHdpdGggdGhlIG5hbWVzIGNvcnJlc3BvbmRpbmdcbiAqICAgdG8gbmFtZXMgaW4gYHNwZWNzYC5cbiAqIEB0aHJvd3MgRXJyb3IsIGlmIGFueSBvZiB0aGUgdGVuc29ycyBoYXMgdW5zdXBwb3J0ZWQgZHR5cGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGVXZWlnaHRzKFxuICAgIGJ1ZmZlcjogQXJyYXlCdWZmZXIsIHNwZWNzOiBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdKTogTmFtZWRUZW5zb3JNYXAge1xuICAvLyBUT0RPKGFkYXJvYiwgY2Fpcyk6IFN1cHBvcnQgcXVhbnRpemF0aW9uLlxuICBjb25zdCBvdXQ6IE5hbWVkVGVuc29yTWFwID0ge307XG4gIGxldCBmbG9hdDE2RGVjb2RlOiAoYnVmZmVyOiBVaW50MTZBcnJheSkgPT4gRmxvYXQzMkFycmF5IHwgdW5kZWZpbmVkO1xuICBsZXQgb2Zmc2V0ID0gMDtcbiAgZm9yIChjb25zdCBzcGVjIG9mIHNwZWNzKSB7XG4gICAgY29uc3QgbmFtZSA9IHNwZWMubmFtZTtcbiAgICBjb25zdCBkdHlwZSA9IHNwZWMuZHR5cGU7XG4gICAgY29uc3Qgc2hhcGUgPSBzcGVjLnNoYXBlO1xuICAgIGNvbnN0IHNpemUgPSBzaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICBsZXQgdmFsdWVzOiBUeXBlZEFycmF5fHN0cmluZ1tdfFVpbnQ4QXJyYXlbXTtcblxuICAgIGlmICgncXVhbnRpemF0aW9uJyBpbiBzcGVjKSB7XG4gICAgICBjb25zdCBxdWFudGl6YXRpb24gPSBzcGVjLnF1YW50aXphdGlvbjtcbiAgICAgIGlmIChxdWFudGl6YXRpb24uZHR5cGUgPT09ICd1aW50OCcgfHwgcXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDE2Jykge1xuICAgICAgICBpZiAoISgnbWluJyBpbiBxdWFudGl6YXRpb24gJiYgJ3NjYWxlJyBpbiBxdWFudGl6YXRpb24pKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICBgV2VpZ2h0ICR7c3BlYy5uYW1lfSB3aXRoIHF1YW50aXphdGlvbiAke3F1YW50aXphdGlvbi5kdHlwZX0gYCArXG4gICAgICAgICAgICAgIGBkb2Vzbid0IGhhdmUgY29ycmVzcG9uZGluZyBtZXRhZGF0YSBtaW4gYW5kIHNjYWxlLmApO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ2Zsb2F0MTYnKSB7XG4gICAgICAgIGlmIChkdHlwZSAhPT0gJ2Zsb2F0MzInKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICBgV2VpZ2h0ICR7c3BlYy5uYW1lfSBpcyBxdWFudGl6ZWQgd2l0aCAke3F1YW50aXphdGlvbi5kdHlwZX0gYCArXG4gICAgICAgICAgICAgIGB3aGljaCBvbmx5IHN1cHBvcnRzIHdlaWdodHMgb2YgdHlwZSBmbG9hdDMyIG5vdCAke2R0eXBlfS5gKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgYFdlaWdodCAke3NwZWMubmFtZX0gaGFzIHVua25vd24gYCArXG4gICAgICAgICAgICBgcXVhbnRpemF0aW9uIGR0eXBlICR7cXVhbnRpemF0aW9uLmR0eXBlfS4gYCArXG4gICAgICAgICAgICBgU3VwcG9ydGVkIHF1YW50aXphdGlvbiBkdHlwZXMgYXJlOiBgICtcbiAgICAgICAgICAgIGAndWludDgnLCAndWludDE2JywgYW5kICdmbG9hdDE2Jy5gKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHF1YW50aXphdGlvblNpemVGYWN0b3IgPSBEVFlQRV9WQUxVRV9TSVpFX01BUFtxdWFudGl6YXRpb24uZHR5cGVdO1xuICAgICAgY29uc3QgYnl0ZUJ1ZmZlciA9XG4gICAgICAgICAgYnVmZmVyLnNsaWNlKG9mZnNldCwgb2Zmc2V0ICsgc2l6ZSAqIHF1YW50aXphdGlvblNpemVGYWN0b3IpO1xuICAgICAgY29uc3QgcXVhbnRpemVkQXJyYXkgPSAocXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDgnKSA/XG4gICAgICAgICAgbmV3IFVpbnQ4QXJyYXkoYnl0ZUJ1ZmZlcikgOlxuICAgICAgICAgIG5ldyBVaW50MTZBcnJheShieXRlQnVmZmVyKTtcbiAgICAgIGlmIChkdHlwZSA9PT0gJ2Zsb2F0MzInKSB7XG4gICAgICAgIGlmIChxdWFudGl6YXRpb24uZHR5cGUgPT09ICd1aW50OCcgfHwgcXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDE2Jykge1xuICAgICAgICAgIHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkocXVhbnRpemVkQXJyYXkubGVuZ3RoKTtcbiAgICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHF1YW50aXplZEFycmF5Lmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICBjb25zdCB2ID0gcXVhbnRpemVkQXJyYXlbaV07XG4gICAgICAgICAgICB2YWx1ZXNbaV0gPSB2ICogcXVhbnRpemF0aW9uLnNjYWxlICsgcXVhbnRpemF0aW9uLm1pbjtcbiAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSBpZiAocXVhbnRpemF0aW9uLmR0eXBlID09PSAnZmxvYXQxNicpIHtcbiAgICAgICAgICBpZiAoZmxvYXQxNkRlY29kZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgICAgICBmbG9hdDE2RGVjb2RlID0gZ2V0RmxvYXQxNkRlY29kZXIoKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdmFsdWVzID0gZmxvYXQxNkRlY29kZShxdWFudGl6ZWRBcnJheSBhcyBVaW50MTZBcnJheSk7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICBgVW5zdXBwb3J0ZWQgcXVhbnRpemF0aW9uIHR5cGUgJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgZm9yIHdlaWdodCB0eXBlIGZsb2F0MzIuYCk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICAgICAgaWYgKHF1YW50aXphdGlvbi5kdHlwZSAhPT0gJ3VpbnQ4JyAmJiBxdWFudGl6YXRpb24uZHR5cGUgIT09ICd1aW50MTYnKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICBgVW5zdXBwb3J0ZWQgcXVhbnRpemF0aW9uIHR5cGUgJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgZm9yIHdlaWdodCB0eXBlIGludDMyLmApO1xuICAgICAgICB9XG4gICAgICAgIHZhbHVlcyA9IG5ldyBJbnQzMkFycmF5KHF1YW50aXplZEFycmF5Lmxlbmd0aCk7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcXVhbnRpemVkQXJyYXkubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICBjb25zdCB2ID0gcXVhbnRpemVkQXJyYXlbaV07XG4gICAgICAgICAgdmFsdWVzW2ldID0gTWF0aC5yb3VuZCh2ICogcXVhbnRpemF0aW9uLnNjYWxlICsgcXVhbnRpemF0aW9uLm1pbik7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZHR5cGUgaW4gd2VpZ2h0ICcke25hbWV9JzogJHtkdHlwZX1gKTtcbiAgICAgIH1cbiAgICAgIG9mZnNldCArPSBzaXplICogcXVhbnRpemF0aW9uU2l6ZUZhY3RvcjtcbiAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3Qgc2l6ZSA9IHNpemVGcm9tU2hhcGUoc3BlYy5zaGFwZSk7XG4gICAgICB2YWx1ZXMgPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgc2l6ZTsgaSsrKSB7XG4gICAgICAgIGNvbnN0IGJ5dGVMZW5ndGggPSBuZXcgVWludDMyQXJyYXkoXG4gICAgICAgICAgICBidWZmZXIuc2xpY2Uob2Zmc2V0LCBvZmZzZXQgKyBOVU1fQllURVNfU1RSSU5HX0xFTkdUSCkpWzBdO1xuICAgICAgICBvZmZzZXQgKz0gTlVNX0JZVEVTX1NUUklOR19MRU5HVEg7XG4gICAgICAgIGNvbnN0IGJ5dGVzID0gbmV3IFVpbnQ4QXJyYXkoYnVmZmVyLnNsaWNlKG9mZnNldCwgb2Zmc2V0ICsgYnl0ZUxlbmd0aCkpO1xuICAgICAgICAodmFsdWVzIGFzIFVpbnQ4QXJyYXlbXSkucHVzaChieXRlcyk7XG4gICAgICAgIG9mZnNldCArPSBieXRlTGVuZ3RoO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBkdHlwZUZhY3RvciA9IERUWVBFX1ZBTFVFX1NJWkVfTUFQW2R0eXBlXTtcbiAgICAgIGNvbnN0IGJ5dGVCdWZmZXIgPSBidWZmZXIuc2xpY2Uob2Zmc2V0LCBvZmZzZXQgKyBzaXplICogZHR5cGVGYWN0b3IpO1xuXG4gICAgICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgICAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGJ5dGVCdWZmZXIpO1xuICAgICAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgICAgICB2YWx1ZXMgPSBuZXcgSW50MzJBcnJheShieXRlQnVmZmVyKTtcbiAgICAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgICAgICB2YWx1ZXMgPSBuZXcgVWludDhBcnJheShieXRlQnVmZmVyKTtcbiAgICAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICAgIHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICAgIGNvbnN0IHJlYWwgPSBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcy5sZW5ndGggLyAyKTtcbiAgICAgICAgY29uc3QgaW1hZ2UgPSBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcy5sZW5ndGggLyAyKTtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZWFsLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgcmVhbFtpXSA9IHZhbHVlc1tpICogMl07XG4gICAgICAgICAgaW1hZ2VbaV0gPSB2YWx1ZXNbaSAqIDIgKyAxXTtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCByZWFsVGVuc29yID0gdGVuc29yKHJlYWwsIHNoYXBlLCAnZmxvYXQzMicpO1xuICAgICAgICBjb25zdCBpbWFnZVRlbnNvciA9IHRlbnNvcihpbWFnZSwgc2hhcGUsICdmbG9hdDMyJyk7XG4gICAgICAgIG91dFtuYW1lXSA9IGNvbXBsZXgocmVhbFRlbnNvciwgaW1hZ2VUZW5zb3IpO1xuICAgICAgICByZWFsVGVuc29yLmRpc3Bvc2UoKTtcbiAgICAgICAgaW1hZ2VUZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBkdHlwZSBpbiB3ZWlnaHQgJyR7bmFtZX0nOiAke2R0eXBlfWApO1xuICAgICAgfVxuICAgICAgb2Zmc2V0ICs9IHNpemUgKiBkdHlwZUZhY3RvcjtcbiAgICB9XG4gICAgaWYgKGR0eXBlICE9PSAnY29tcGxleDY0Jykge1xuICAgICAgb3V0W25hbWVdID0gdGVuc29yKHZhbHVlcywgc2hhcGUsIGR0eXBlKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIG91dDtcbn1cblxuLyoqXG4gKiBDb25jYXRlbmF0ZSBUeXBlZEFycmF5cyBpbnRvIGFuIEFycmF5QnVmZmVyLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29uY2F0ZW5hdGVUeXBlZEFycmF5cyh4czogVHlwZWRBcnJheVtdKTogQXJyYXlCdWZmZXIge1xuICAvLyBUT0RPKGFkYXJvYiwgY2Fpcyk6IFN1cHBvcnQgcXVhbnRpemF0aW9uLlxuICBpZiAoeHMgPT09IG51bGwpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYEludmFsaWQgaW5wdXQgdmFsdWU6ICR7SlNPTi5zdHJpbmdpZnkoeHMpfWApO1xuICB9XG5cbiAgbGV0IHRvdGFsQnl0ZUxlbmd0aCA9IDA7XG5cbiAgLy8gYG5vcm1hbGl6ZWRYc2AgaXMgaGVyZSBmb3IgdGhpcyByZWFzb246IGEgYFR5cGVkQXJyYXlgJ3MgYGJ1ZmZlcidcbiAgLy8gY2FuIGhhdmUgYSBkaWZmZXJlbnQgYnl0ZSBsZW5ndGggZnJvbSB0aGF0IG9mIHRoZSBgVHlwZWRBcnJheWAgaXRzZWxmLFxuICAvLyBmb3IgZXhhbXBsZSwgd2hlbiB0aGUgYFR5cGVkQXJyYXlgIGlzIGNyZWF0ZWQgZnJvbSBhbiBvZmZzZXQgaW4gYW5cbiAgLy8gYEFycmF5QnVmZmVyYC4gYG5vcm1saWF6ZWRYc2AgaG9sZHMgYFR5cGVkQXJyYXlgcyB3aG9zZSBgYnVmZmVyYHMgbWF0Y2hcbiAgLy8gdGhlIGBUeXBlZEFycmF5YCBpbiBieXRlIGxlbmd0aC4gSWYgYW4gZWxlbWVudCBvZiBgeHNgIGRvZXMgbm90IHNob3dcbiAgLy8gdGhpcyBwcm9wZXJ0eSwgYSBuZXcgYFR5cGVkQXJyYXlgIHRoYXQgc2F0aXNmeSB0aGlzIHByb3BlcnR5IHdpbGwgYmVcbiAgLy8gY29uc3RydWN0ZWQgYW5kIHB1c2hlZCBpbnRvIGBub3JtYWxpemVkWHNgLlxuICBjb25zdCBub3JtYWxpemVkWHM6IFR5cGVkQXJyYXlbXSA9IFtdO1xuICB4cy5mb3JFYWNoKCh4OiBUeXBlZEFycmF5KSA9PiB7XG4gICAgdG90YWxCeXRlTGVuZ3RoICs9IHguYnl0ZUxlbmd0aDtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZTpuby1hbnlcbiAgICBub3JtYWxpemVkWHMucHVzaChcbiAgICAgICAgeC5ieXRlTGVuZ3RoID09PSB4LmJ1ZmZlci5ieXRlTGVuZ3RoID8geCA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldyAoeC5jb25zdHJ1Y3RvciBhcyBhbnkpKHgpKTtcbiAgICBpZiAoISh4IGFzIGFueSBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSB8fCB4IGFzIGFueSBpbnN0YW5jZW9mIEludDMyQXJyYXkgfHxcbiAgICAgICAgICB4IGFzIGFueSBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkpKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIFR5cGVkQXJyYXkgc3VidHlwZTogJHt4LmNvbnN0cnVjdG9yLm5hbWV9YCk7XG4gICAgfVxuICAgIC8vIHRzbGludDplbmFibGU6bm8tYW55XG4gIH0pO1xuXG4gIGNvbnN0IHkgPSBuZXcgVWludDhBcnJheSh0b3RhbEJ5dGVMZW5ndGgpO1xuICBsZXQgb2Zmc2V0ID0gMDtcbiAgbm9ybWFsaXplZFhzLmZvckVhY2goKHg6IFR5cGVkQXJyYXkpID0+IHtcbiAgICB5LnNldChuZXcgVWludDhBcnJheSh4LmJ1ZmZlciksIG9mZnNldCk7XG4gICAgb2Zmc2V0ICs9IHguYnl0ZUxlbmd0aDtcbiAgfSk7XG5cbiAgcmV0dXJuIHkuYnVmZmVyO1xufVxuXG4vLyBVc2UgQnVmZmVyIG9uIE5vZGUuanMgaW5zdGVhZCBvZiBCbG9iL2F0b2IvYnRvYVxuY29uc3QgdXNlTm9kZUJ1ZmZlciA9IHR5cGVvZiBCdWZmZXIgIT09ICd1bmRlZmluZWQnICYmXG4gICAgKHR5cGVvZiBCbG9iID09PSAndW5kZWZpbmVkJyB8fCB0eXBlb2YgYXRvYiA9PT0gJ3VuZGVmaW5lZCcgfHxcbiAgICAgdHlwZW9mIGJ0b2EgPT09ICd1bmRlZmluZWQnKTtcblxuLyoqXG4gKiBDYWxjdWxhdGUgdGhlIGJ5dGUgbGVuZ3RoIG9mIGEgSmF2YVNjcmlwdCBzdHJpbmcuXG4gKlxuICogTm90ZSB0aGF0IGEgSmF2YVNjcmlwdCBzdHJpbmcgY2FuIGNvbnRhaW4gd2lkZSBjaGFyYWN0ZXJzLCB0aGVyZWZvcmUgdGhlXG4gKiBsZW5ndGggb2YgdGhlIHN0cmluZyBpcyBub3QgbmVjZXNzYXJpbHkgZXF1YWwgdG8gdGhlIGJ5dGUgbGVuZ3RoLlxuICpcbiAqIEBwYXJhbSBzdHIgSW5wdXQgc3RyaW5nLlxuICogQHJldHVybnMgQnl0ZSBsZW5ndGguXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzdHJpbmdCeXRlTGVuZ3RoKHN0cjogc3RyaW5nKTogbnVtYmVyIHtcbiAgaWYgKHVzZU5vZGVCdWZmZXIpIHtcbiAgICByZXR1cm4gQnVmZmVyLmJ5dGVMZW5ndGgoc3RyKTtcbiAgfVxuICByZXR1cm4gbmV3IEJsb2IoW3N0cl0pLnNpemU7XG59XG5cbi8qKlxuICogRW5jb2RlIGFuIEFycmF5QnVmZmVyIGFzIGEgYmFzZTY0IGVuY29kZWQgc3RyaW5nLlxuICpcbiAqIEBwYXJhbSBidWZmZXIgYEFycmF5QnVmZmVyYCB0byBiZSBjb252ZXJ0ZWQuXG4gKiBAcmV0dXJucyBBIHN0cmluZyB0aGF0IGJhc2U2NC1lbmNvZGVzIGBidWZmZXJgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gYXJyYXlCdWZmZXJUb0Jhc2U2NFN0cmluZyhidWZmZXI6IEFycmF5QnVmZmVyKTogc3RyaW5nIHtcbiAgaWYgKHVzZU5vZGVCdWZmZXIpIHtcbiAgICByZXR1cm4gQnVmZmVyLmZyb20oYnVmZmVyKS50b1N0cmluZygnYmFzZTY0Jyk7XG4gIH1cbiAgY29uc3QgYnVmID0gbmV3IFVpbnQ4QXJyYXkoYnVmZmVyKTtcbiAgbGV0IHMgPSAnJztcbiAgZm9yIChsZXQgaSA9IDAsIGwgPSBidWYubGVuZ3RoOyBpIDwgbDsgaSsrKSB7XG4gICAgcyArPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGJ1ZltpXSk7XG4gIH1cbiAgcmV0dXJuIGJ0b2Eocyk7XG59XG5cbi8qKlxuICogRGVjb2RlIGEgYmFzZTY0IHN0cmluZyBhcyBhbiBBcnJheUJ1ZmZlci5cbiAqXG4gKiBAcGFyYW0gc3RyIEJhc2U2NCBzdHJpbmcuXG4gKiBAcmV0dXJucyBEZWNvZGVkIGBBcnJheUJ1ZmZlcmAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBiYXNlNjRTdHJpbmdUb0FycmF5QnVmZmVyKHN0cjogc3RyaW5nKTogQXJyYXlCdWZmZXIge1xuICBpZiAodXNlTm9kZUJ1ZmZlcikge1xuICAgIGNvbnN0IGJ1ZiA9IEJ1ZmZlci5mcm9tKHN0ciwgJ2Jhc2U2NCcpO1xuICAgIHJldHVybiBidWYuYnVmZmVyLnNsaWNlKGJ1Zi5ieXRlT2Zmc2V0LCBidWYuYnl0ZU9mZnNldCArIGJ1Zi5ieXRlTGVuZ3RoKTtcbiAgfVxuICBjb25zdCBzID0gYXRvYihzdHIpO1xuICBjb25zdCBidWZmZXIgPSBuZXcgVWludDhBcnJheShzLmxlbmd0aCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcy5sZW5ndGg7ICsraSkge1xuICAgIGJ1ZmZlci5zZXQoW3MuY2hhckNvZGVBdChpKV0sIGkpO1xuICB9XG4gIHJldHVybiBidWZmZXIuYnVmZmVyO1xufVxuXG4vKipcbiAqIENvbmNhdGVuYXRlIGEgbnVtYmVyIG9mIEFycmF5QnVmZmVycyBpbnRvIG9uZS5cbiAqXG4gKiBAcGFyYW0gYnVmZmVycyBBIG51bWJlciBvZiBhcnJheSBidWZmZXJzIHRvIGNvbmNhdGVuYXRlLlxuICogQHJldHVybnMgUmVzdWx0IG9mIGNvbmNhdGVuYXRpbmcgYGJ1ZmZlcnNgIGluIG9yZGVyLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29uY2F0ZW5hdGVBcnJheUJ1ZmZlcnMoYnVmZmVyczogQXJyYXlCdWZmZXJbXSk6IEFycmF5QnVmZmVyIHtcbiAgaWYgKGJ1ZmZlcnMubGVuZ3RoID09PSAxKSB7XG4gICAgcmV0dXJuIGJ1ZmZlcnNbMF07XG4gIH1cblxuICBsZXQgdG90YWxCeXRlTGVuZ3RoID0gMDtcbiAgYnVmZmVycy5mb3JFYWNoKChidWZmZXI6IEFycmF5QnVmZmVyKSA9PiB7XG4gICAgdG90YWxCeXRlTGVuZ3RoICs9IGJ1ZmZlci5ieXRlTGVuZ3RoO1xuICB9KTtcblxuICBjb25zdCB0ZW1wID0gbmV3IFVpbnQ4QXJyYXkodG90YWxCeXRlTGVuZ3RoKTtcbiAgbGV0IG9mZnNldCA9IDA7XG4gIGJ1ZmZlcnMuZm9yRWFjaCgoYnVmZmVyOiBBcnJheUJ1ZmZlcikgPT4ge1xuICAgIHRlbXAuc2V0KG5ldyBVaW50OEFycmF5KGJ1ZmZlciksIG9mZnNldCk7XG4gICAgb2Zmc2V0ICs9IGJ1ZmZlci5ieXRlTGVuZ3RoO1xuICB9KTtcbiAgcmV0dXJuIHRlbXAuYnVmZmVyO1xufVxuXG4vKipcbiAqIEdldCB0aGUgYmFzZW5hbWUgb2YgYSBwYXRoLlxuICpcbiAqIEJlaGF2ZXMgaW4gYSB3YXkgYW5hbG9nb3VzIHRvIExpbnV4J3MgYmFzZW5hbWUgY29tbWFuZC5cbiAqXG4gKiBAcGFyYW0gcGF0aFxuICovXG5leHBvcnQgZnVuY3Rpb24gYmFzZW5hbWUocGF0aDogc3RyaW5nKTogc3RyaW5nIHtcbiAgY29uc3QgU0VQQVJBVE9SID0gJy8nO1xuICBwYXRoID0gcGF0aC50cmltKCk7XG4gIHdoaWxlIChwYXRoLmVuZHNXaXRoKFNFUEFSQVRPUikpIHtcbiAgICBwYXRoID0gcGF0aC5zbGljZSgwLCBwYXRoLmxlbmd0aCAtIDEpO1xuICB9XG4gIGNvbnN0IGl0ZW1zID0gcGF0aC5zcGxpdChTRVBBUkFUT1IpO1xuICByZXR1cm4gaXRlbXNbaXRlbXMubGVuZ3RoIC0gMV07XG59XG5cbi8qKlxuICogQ3JlYXRlIGBNb2RlbEpTT05gIGZyb20gYE1vZGVsQXJ0aWZhY3RzYC5cbiAqXG4gKiBAcGFyYW0gYXJ0aWZhY3RzIE1vZGVsIGFydGlmYWN0cywgZGVzY3JpYmluZyB0aGUgbW9kZWwgYW5kIGl0cyB3ZWlnaHRzLlxuICogQHBhcmFtIG1hbmlmZXN0IFdlaWdodCBtYW5pZmVzdCwgZGVzY3JpYmluZyB3aGVyZSB0aGUgd2VpZ2h0cyBvZiB0aGVcbiAqICAgICBgTW9kZWxBcnRpZmFjdHNgIGFyZSBzdG9yZWQsIGFuZCBzb21lIG1ldGFkYXRhIGFib3V0IHRoZW0uXG4gKiBAcmV0dXJucyBPYmplY3QgcmVwcmVzZW50aW5nIHRoZSBgbW9kZWwuanNvbmAgZmlsZSBkZXNjcmliaW5nIHRoZSBtb2RlbFxuICogICAgIGFydGlmYWN0cyBhbmQgd2VpZ2h0c1xuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0TW9kZWxKU09ORm9yTW9kZWxBcnRpZmFjdHMoXG4gICAgYXJ0aWZhY3RzOiBNb2RlbEFydGlmYWN0cywgbWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZyk6IE1vZGVsSlNPTiB7XG4gIGNvbnN0IHJlc3VsdDogTW9kZWxKU09OID0ge1xuICAgIG1vZGVsVG9wb2xvZ3k6IGFydGlmYWN0cy5tb2RlbFRvcG9sb2d5LFxuICAgIGZvcm1hdDogYXJ0aWZhY3RzLmZvcm1hdCxcbiAgICBnZW5lcmF0ZWRCeTogYXJ0aWZhY3RzLmdlbmVyYXRlZEJ5LFxuICAgIGNvbnZlcnRlZEJ5OiBhcnRpZmFjdHMuY29udmVydGVkQnksXG4gICAgd2VpZ2h0c01hbmlmZXN0OiBtYW5pZmVzdFxuICB9O1xuICBpZiAoYXJ0aWZhY3RzLnNpZ25hdHVyZSAhPSBudWxsKSB7XG4gICAgcmVzdWx0LnNpZ25hdHVyZSA9IGFydGlmYWN0cy5zaWduYXR1cmU7XG4gIH1cbiAgaWYgKGFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhICE9IG51bGwpIHtcbiAgICByZXN1bHQudXNlckRlZmluZWRNZXRhZGF0YSA9IGFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhO1xuICB9XG4gIGlmIChhcnRpZmFjdHMubW9kZWxJbml0aWFsaXplciAhPSBudWxsKSB7XG4gICAgcmVzdWx0Lm1vZGVsSW5pdGlhbGl6ZXIgPSBhcnRpZmFjdHMubW9kZWxJbml0aWFsaXplcjtcbiAgfVxuICBpZiAoYXJ0aWZhY3RzLmluaXRpYWxpemVyU2lnbmF0dXJlICE9IG51bGwpIHtcbiAgICByZXN1bHQuaW5pdGlhbGl6ZXJTaWduYXR1cmUgPSBhcnRpZmFjdHMuaW5pdGlhbGl6ZXJTaWduYXR1cmU7XG4gIH1cbiAgaWYgKGFydGlmYWN0cy50cmFpbmluZ0NvbmZpZyAhPSBudWxsKSB7XG4gICAgcmVzdWx0LnRyYWluaW5nQ29uZmlnID0gYXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnO1xuICB9XG4gIHJldHVybiByZXN1bHQ7XG59XG5cbi8qKlxuICogQ3JlYXRlIGBNb2RlbEFydGlmYWN0c2AgZnJvbSBhIEpTT04gZmlsZSBhbmQgd2VpZ2h0cy5cbiAqXG4gKiBAcGFyYW0gbW9kZWxKU09OIE9iamVjdCBjb250YWluaW5nIHRoZSBwYXJzZWQgSlNPTiBvZiBgbW9kZWwuanNvbmBcbiAqIEBwYXJhbSB3ZWlnaHRTcGVjcyBUaGUgbGlzdCBvZiBXZWlnaHRzTWFuaWZlc3RFbnRyeSBmb3IgdGhlIG1vZGVsLiBNdXN0IGJlXG4gKiAgICAgcGFzc2VkIGlmIHRoZSBtb2RlbEpTT04gaGFzIGEgd2VpZ2h0c01hbmlmZXN0LlxuICogQHBhcmFtIHdlaWdodERhdGEgQW4gQXJyYXlCdWZmZXIgb2Ygd2VpZ2h0IGRhdGEgZm9yIHRoZSBtb2RlbCBjb3JyZXNwb25kaW5nXG4gKiAgICAgdG8gdGhlIHdlaWdodHMgaW4gd2VpZ2h0U3BlY3MuIE11c3QgYmUgcGFzc2VkIGlmIHRoZSBtb2RlbEpTT04gaGFzIGFcbiAqICAgICB3ZWlnaHRzTWFuaWZlc3QuXG4gKiBAcmV0dXJucyBBIFByb21pc2Ugb2YgdGhlIGBNb2RlbEFydGlmYWN0c2AsIGFzIGRlc2NyaWJlZCBieSB0aGUgSlNPTiBmaWxlLlxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OU3luYyhcbiAgICBtb2RlbEpTT046IE1vZGVsSlNPTiwgd2VpZ2h0U3BlY3M/OiBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdLFxuICAgIHdlaWdodERhdGE/OiBBcnJheUJ1ZmZlcik6IE1vZGVsQXJ0aWZhY3RzIHtcblxuICBjb25zdCBtb2RlbEFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMgPSB7XG4gICAgbW9kZWxUb3BvbG9neTogbW9kZWxKU09OLm1vZGVsVG9wb2xvZ3ksXG4gICAgZm9ybWF0OiBtb2RlbEpTT04uZm9ybWF0LFxuICAgIGdlbmVyYXRlZEJ5OiBtb2RlbEpTT04uZ2VuZXJhdGVkQnksXG4gICAgY29udmVydGVkQnk6IG1vZGVsSlNPTi5jb252ZXJ0ZWRCeVxuICB9O1xuXG4gIGlmIChtb2RlbEpTT04udHJhaW5pbmdDb25maWcgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnID0gbW9kZWxKU09OLnRyYWluaW5nQ29uZmlnO1xuICB9XG4gIGlmIChtb2RlbEpTT04ud2VpZ2h0c01hbmlmZXN0ICE9IG51bGwpIHtcbiAgICBpZiAoIXdlaWdodFNwZWNzKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ21vZGVsSlNPTiBoYXMgd2VpZ2h0c01hbmlmZXN0IGJ1dCB3ZWlnaHRTcGVjcyBpcyBudWxsJyk7XG4gICAgfVxuICAgIGlmICghd2VpZ2h0RGF0YSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdtb2RlbEpTT04gaGFzIHdlaWdodHNNYW5pZmVzdCBidXQgd2VpZ2h0RGF0YSBpcyBudWxsJyk7XG4gICAgfVxuICAgIG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzID0gd2VpZ2h0U3BlY3M7XG4gICAgbW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSA9IHdlaWdodERhdGE7XG4gIH1cbiAgaWYgKG1vZGVsSlNPTi5zaWduYXR1cmUgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLnNpZ25hdHVyZSA9IG1vZGVsSlNPTi5zaWduYXR1cmU7XG4gIH1cbiAgaWYgKG1vZGVsSlNPTi51c2VyRGVmaW5lZE1ldGFkYXRhICE9IG51bGwpIHtcbiAgICBtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhID0gbW9kZWxKU09OLnVzZXJEZWZpbmVkTWV0YWRhdGE7XG4gIH1cbiAgaWYgKG1vZGVsSlNPTi5tb2RlbEluaXRpYWxpemVyICE9IG51bGwpIHtcbiAgICBtb2RlbEFydGlmYWN0cy5tb2RlbEluaXRpYWxpemVyID0gbW9kZWxKU09OLm1vZGVsSW5pdGlhbGl6ZXI7XG4gIH1cbiAgaWYgKG1vZGVsSlNPTi5pbml0aWFsaXplclNpZ25hdHVyZSAhPSBudWxsKSB7XG4gICAgbW9kZWxBcnRpZmFjdHMuaW5pdGlhbGl6ZXJTaWduYXR1cmUgPSBtb2RlbEpTT04uaW5pdGlhbGl6ZXJTaWduYXR1cmU7XG4gIH1cblxuICByZXR1cm4gbW9kZWxBcnRpZmFjdHM7XG59XG5cbi8qKlxuICogQ3JlYXRlIGBNb2RlbEFydGlmYWN0c2AgZnJvbSBhIEpTT04gZmlsZS5cbiAqXG4gKiBAcGFyYW0gbW9kZWxKU09OIE9iamVjdCBjb250YWluaW5nIHRoZSBwYXJzZWQgSlNPTiBvZiBgbW9kZWwuanNvbmBcbiAqIEBwYXJhbSBsb2FkV2VpZ2h0cyBGdW5jdGlvbiB0aGF0IHRha2VzIHRoZSBKU09OIGZpbGUncyB3ZWlnaHRzIG1hbmlmZXN0LFxuICogICAgIHJlYWRzIHdlaWdodHMgZnJvbSB0aGUgbGlzdGVkIHBhdGgocyksIGFuZCByZXR1cm5zIGEgUHJvbWlzZSBvZiB0aGVcbiAqICAgICB3ZWlnaHQgbWFuaWZlc3QgZW50cmllcyBhbG9uZyB3aXRoIHRoZSB3ZWlnaHRzIGRhdGEuXG4gKiBAcmV0dXJucyBBIFByb21pc2Ugb2YgdGhlIGBNb2RlbEFydGlmYWN0c2AsIGFzIGRlc2NyaWJlZCBieSB0aGUgSlNPTiBmaWxlLlxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OKFxuICAgIG1vZGVsSlNPTjogTW9kZWxKU09OLFxuICAgIGxvYWRXZWlnaHRzOiAod2VpZ2h0c01hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcpID0+IFByb21pc2U8W1xuICAgICAgLyogd2VpZ2h0U3BlY3MgKi8gV2VpZ2h0c01hbmlmZXN0RW50cnlbXSwgLyogd2VpZ2h0RGF0YSAqLyBBcnJheUJ1ZmZlclxuICAgIF0+KTogUHJvbWlzZTxNb2RlbEFydGlmYWN0cz4ge1xuICBsZXQgd2VpZ2h0U3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10gfCB1bmRlZmluZWQ7XG4gIGxldCB3ZWlnaHREYXRhOiBBcnJheUJ1ZmZlciB8IHVuZGVmaW5lZDtcblxuICBpZiAobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdCAhPSBudWxsKSB7XG4gICAgW3dlaWdodFNwZWNzLCB3ZWlnaHREYXRhXSA9IGF3YWl0IGxvYWRXZWlnaHRzKG1vZGVsSlNPTi53ZWlnaHRzTWFuaWZlc3QpO1xuICB9XG5cbiAgcmV0dXJuIGdldE1vZGVsQXJ0aWZhY3RzRm9ySlNPTlN5bmMobW9kZWxKU09OLCB3ZWlnaHRTcGVjcywgd2VpZ2h0RGF0YSk7XG59XG5cbi8qKlxuICogUG9wdWxhdGUgTW9kZWxBcnRpZmFjdHNJbmZvIGZpZWxkcyBmb3IgYSBtb2RlbCB3aXRoIEpTT04gdG9wb2xvZ3kuXG4gKiBAcGFyYW0gbW9kZWxBcnRpZmFjdHNcbiAqIEByZXR1cm5zIEEgTW9kZWxBcnRpZmFjdHNJbmZvIG9iamVjdC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldE1vZGVsQXJ0aWZhY3RzSW5mb0ZvckpTT04obW9kZWxBcnRpZmFjdHM6IE1vZGVsQXJ0aWZhY3RzKTpcbiAgICBNb2RlbEFydGlmYWN0c0luZm8ge1xuICBpZiAobW9kZWxBcnRpZmFjdHMubW9kZWxUb3BvbG9neSBpbnN0YW5jZW9mIEFycmF5QnVmZmVyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdFeHBlY3RlZCBKU09OIG1vZGVsIHRvcG9sb2d5LCByZWNlaXZlZCBBcnJheUJ1ZmZlci4nKTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgZGF0ZVNhdmVkOiBuZXcgRGF0ZSgpLFxuICAgIG1vZGVsVG9wb2xvZ3lUeXBlOiAnSlNPTicsXG4gICAgbW9kZWxUb3BvbG9neUJ5dGVzOiBtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5ID09IG51bGwgP1xuICAgICAgICAwIDpcbiAgICAgICAgc3RyaW5nQnl0ZUxlbmd0aChKU09OLnN0cmluZ2lmeShtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KSksXG4gICAgd2VpZ2h0U3BlY3NCeXRlczogbW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MgPT0gbnVsbCA/XG4gICAgICAgIDAgOlxuICAgICAgICBzdHJpbmdCeXRlTGVuZ3RoKEpTT04uc3RyaW5naWZ5KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKSksXG4gICAgd2VpZ2h0RGF0YUJ5dGVzOiBtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhID09IG51bGwgP1xuICAgICAgICAwIDpcbiAgICAgICAgbW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YS5ieXRlTGVuZ3RoLFxuICB9O1xufVxuXG4vKipcbiAqIENvbmNhdGVuYXRlIHRoZSB3ZWlnaHRzIHN0b3JlZCBpbiBhIFdlaWdodHNNYW5pZmVzdENvbmZpZyBpbnRvIGEgbGlzdCBvZlxuICogV2VpZ2h0c01hbmlmZXN0RW50cnlcbiAqXG4gKiBAcGFyYW0gd2VpZ2h0c01hbmlmZXN0IFRoZSBXZWlnaHRzTWFuaWZlc3RDb25maWcgdG8gZXh0cmFjdCB3ZWlnaHRzIGZyb20uXG4gKiBAcmV0dXJucyBBIGxpc3Qgb2YgV2VpZ2h0c01hbmlmZXN0RW50cnkgb2YgdGhlIHdlaWdodHMgaW4gdGhlIHdlaWdodHNNYW5pZmVzdFxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2VpZ2h0U3BlY3Mod2VpZ2h0c01hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcpOlxuICAgIFdlaWdodHNNYW5pZmVzdEVudHJ5W10ge1xuICBjb25zdCB3ZWlnaHRTcGVjczogV2VpZ2h0c01hbmlmZXN0RW50cnlbXSA9IFtdO1xuICBmb3IgKGNvbnN0IGVudHJ5IG9mIHdlaWdodHNNYW5pZmVzdCkge1xuICAgIHdlaWdodFNwZWNzLnB1c2goLi4uZW50cnkud2VpZ2h0cyk7XG4gIH1cbiAgcmV0dXJuIHdlaWdodFNwZWNzO1xufVxuXG4vKipcbiAqIENvbXB1dGVzIG1hbnRpc2EgdGFibGUgZm9yIGNhc3RpbmcgRmxvYXQxNiB0byBGbG9hdDMyXG4gKiBTZWUgaHR0cDovL3d3dy5mb3gtdG9vbGtpdC5vcmcvZnRwL2Zhc3RoYWxmZmxvYXRjb252ZXJzaW9uLnBkZlxuICpcbiAqIEByZXR1cm5zIFVpbnQzMkFycmF5LCAyMDQ4IG1hbnRpc3NhIGxvb2t1cCB2YWx1ZXMuXG4gKi9cbmZ1bmN0aW9uIGNvbXB1dGVGbG9hdDE2TWFudGlzYVRhYmxlKCk6IFVpbnQzMkFycmF5IHtcbiAgY29uc3QgY29udmVydE1hbnRpc3NhID0gKGk6IG51bWJlcik6IG51bWJlciA9PiB7XG4gICAgbGV0IG0gPSBpIDw8IDEzO1xuICAgIGxldCBlID0gMDtcblxuICAgIHdoaWxlICgobSAmIDB4MDA4MDAwMDApID09PSAwKSB7XG4gICAgICBlIC09IDB4MDA4MDAwMDA7XG4gICAgICBtIDw8PSAxO1xuICAgIH1cbiAgICBtICY9IH4weDAwODAwMDAwO1xuICAgIGUgKz0gMHgzODgwMDAwMDtcblxuICAgIHJldHVybiBtIHwgZTtcbiAgfTtcblxuICBjb25zdCBtYW50aXNhVGFibGUgPSBuZXcgVWludDMyQXJyYXkoMjA0OCk7XG5cbiAgbWFudGlzYVRhYmxlWzBdID0gMDtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCAxMDI0OyBpKyspIHtcbiAgICBtYW50aXNhVGFibGVbaV0gPSBjb252ZXJ0TWFudGlzc2EoaSk7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDEwMjQ7IGkgPCAyMDQ4OyBpKyspIHtcbiAgICBtYW50aXNhVGFibGVbaV0gPSAweDM4MDAwMDAwICsgKChpIC0gMTAyNCkgPDwgMTMpO1xuICB9XG5cbiAgcmV0dXJuIG1hbnRpc2FUYWJsZTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBleHBvbmVudCB0YWJsZSBmb3IgY2FzdGluZyBGbG9hdDE2IHRvIEZsb2F0MzJcbiAqIFNlZSBodHRwOi8vd3d3LmZveC10b29sa2l0Lm9yZy9mdHAvZmFzdGhhbGZmbG9hdGNvbnZlcnNpb24ucGRmXG4gKlxuICogQHJldHVybnMgVWludDMyQXJyYXksIDY0IGV4cG9uZW50IGxvb2t1cCB2YWx1ZXMuXG4gKi9cbmZ1bmN0aW9uIGNvbXB1dGVGbG9hdDE2RXhwb25lbnRUYWJsZSgpOiBVaW50MzJBcnJheSB7XG4gIGNvbnN0IGV4cG9uZW50VGFibGUgPSBuZXcgVWludDMyQXJyYXkoNjQpO1xuXG4gIGV4cG9uZW50VGFibGVbMF0gPSAwO1xuICBleHBvbmVudFRhYmxlWzMxXSA9IDB4NDc4MDAwMDA7XG4gIGV4cG9uZW50VGFibGVbMzJdID0gMHg4MDAwMDAwMDtcbiAgZXhwb25lbnRUYWJsZVs2M10gPSAweGM3ODAwMDAwO1xuICBmb3IgKGxldCBpID0gMTsgaSA8IDMxOyBpKyspIHtcbiAgICBleHBvbmVudFRhYmxlW2ldID0gaSA8PCAyMztcbiAgfVxuICBmb3IgKGxldCBpID0gMzM7IGkgPCA2MzsgaSsrKSB7XG4gICAgZXhwb25lbnRUYWJsZVtpXSA9IDB4ODAwMDAwMDAgKyAoKGkgLSAzMikgPDwgMjMpO1xuICB9XG5cbiAgcmV0dXJuIGV4cG9uZW50VGFibGU7XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgb2Zmc2V0IHRhYmxlIGZvciBjYXN0aW5nIEZsb2F0MTYgdG8gRmxvYXQzMlxuICogU2VlIGh0dHA6Ly93d3cuZm94LXRvb2xraXQub3JnL2Z0cC9mYXN0aGFsZmZsb2F0Y29udmVyc2lvbi5wZGZcbiAqXG4gKiBAcmV0dXJucyBVaW50MzJBcnJheSwgNmQgb2Zmc2V0IHZhbHVlcy5cbiAqL1xuZnVuY3Rpb24gY29tcHV0ZUZsb2F0MTZPZmZzZXRUYWJsZSgpOiBVaW50MzJBcnJheSB7XG4gIGNvbnN0IG9mZnNldFRhYmxlID0gbmV3IFVpbnQzMkFycmF5KDY0KTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IDY0OyBpKyspIHtcbiAgICBvZmZzZXRUYWJsZVtpXSA9IDEwMjQ7XG4gIH1cbiAgb2Zmc2V0VGFibGVbMF0gPSBvZmZzZXRUYWJsZVszMl0gPSAwO1xuXG4gIHJldHVybiBvZmZzZXRUYWJsZTtcbn1cblxuLyoqXG4gKiBSZXRyaWV2ZSBhIEZsb2F0MTYgZGVjb2RlciB3aGljaCB3aWxsIGRlY29kZSBhIEJ5dGVBcnJheSBvZiBGbG9hdDE2IHZhbHVlc1xuICogdG8gYSBGbG9hdDMyQXJyYXkuXG4gKlxuICogQHJldHVybnMgRnVuY3Rpb24gKGJ1ZmZlcjogVWludDE2QXJyYXkpID0+IEZsb2F0MzJBcnJheSB3aGljaCBkZWNvZGVzXG4gKiAgICAgICAgICB0aGUgVWludDE2QXJyYXkgb2YgRmxvYXQxNiBieXRlcyB0byBhIEZsb2F0MzJBcnJheS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldEZsb2F0MTZEZWNvZGVyKCk6IChidWZmZXI6IFVpbnQxNkFycmF5KSA9PiBGbG9hdDMyQXJyYXkge1xuICAvLyBBbGdvcml0aG0gaXMgYmFzZWQgb2ZmIG9mXG4gIC8vIGh0dHA6Ly93d3cuZm94LXRvb2xraXQub3JnL2Z0cC9mYXN0aGFsZmZsb2F0Y29udmVyc2lvbi5wZGZcblxuICAvLyBDYWNoZSBsb29rdXAgdGFibGVzXG4gIGNvbnN0IG1hbnRpc2FUYWJsZSA9IGNvbXB1dGVGbG9hdDE2TWFudGlzYVRhYmxlKCk7XG4gIGNvbnN0IGV4cG9uZW50VGFibGUgPSBjb21wdXRlRmxvYXQxNkV4cG9uZW50VGFibGUoKTtcbiAgY29uc3Qgb2Zmc2V0VGFibGUgPSBjb21wdXRlRmxvYXQxNk9mZnNldFRhYmxlKCk7XG5cbiAgcmV0dXJuIChxdWFudGl6ZWRBcnJheTogVWludDE2QXJyYXkpID0+IHtcbiAgICBjb25zdCBidWZmZXIgPSBuZXcgQXJyYXlCdWZmZXIoNCAqIHF1YW50aXplZEFycmF5Lmxlbmd0aCk7XG4gICAgY29uc3QgYnVmZmVyVWludDMyVmlldyA9IG5ldyBVaW50MzJBcnJheShidWZmZXIpO1xuICAgIGZvciAobGV0IGluZGV4ID0gMDsgaW5kZXggPCBxdWFudGl6ZWRBcnJheS5sZW5ndGg7IGluZGV4KyspIHtcbiAgICAgIGNvbnN0IGZsb2F0MTZCaXRzID0gcXVhbnRpemVkQXJyYXlbaW5kZXhdO1xuICAgICAgY29uc3QgZmxvYXQzMkJpdHMgPVxuICAgICAgICAgIG1hbnRpc2FUYWJsZVtvZmZzZXRUYWJsZVtmbG9hdDE2Qml0cyA+PiAxMF0gKyAoZmxvYXQxNkJpdHMgJiAweDNmZildICtcbiAgICAgICAgICBleHBvbmVudFRhYmxlW2Zsb2F0MTZCaXRzID4+IDEwXTtcbiAgICAgIGJ1ZmZlclVpbnQzMlZpZXdbaW5kZXhdID0gZmxvYXQzMkJpdHM7XG4gICAgfVxuICAgIHJldHVybiBuZXcgRmxvYXQzMkFycmF5KGJ1ZmZlcik7XG4gIH07XG59XG4iXX0=