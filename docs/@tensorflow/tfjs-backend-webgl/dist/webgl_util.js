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
import { env, util } from '@tensorflow/tfjs-core';
import { getWebGLContext } from './canvas_util';
import { getTextureConfig } from './tex_util';
export function callAndCheck(gl, func) {
    const returnValue = func();
    if (env().getBool('DEBUG')) {
        checkWebGLError(gl);
    }
    return returnValue;
}
function checkWebGLError(gl) {
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
        throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    }
}
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
const MIN_FLOAT16 = 5.96e-8;
const MAX_FLOAT16 = 65504;
export function canBeRepresented(num) {
    if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') || num === 0 ||
        (MIN_FLOAT16 < Math.abs(num) && Math.abs(num) < MAX_FLOAT16)) {
        return true;
    }
    return false;
}
export function getWebGLErrorMessage(gl, status) {
    switch (status) {
        case gl.NO_ERROR:
            return 'NO_ERROR';
        case gl.INVALID_ENUM:
            return 'INVALID_ENUM';
        case gl.INVALID_VALUE:
            return 'INVALID_VALUE';
        case gl.INVALID_OPERATION:
            return 'INVALID_OPERATION';
        case gl.INVALID_FRAMEBUFFER_OPERATION:
            return 'INVALID_FRAMEBUFFER_OPERATION';
        case gl.OUT_OF_MEMORY:
            return 'OUT_OF_MEMORY';
        case gl.CONTEXT_LOST_WEBGL:
            return 'CONTEXT_LOST_WEBGL';
        default:
            return `Unknown error code ${status}`;
    }
}
export function getExtensionOrThrow(gl, extensionName) {
    return throwIfNull(gl, () => gl.getExtension(extensionName), 'Extension "' + extensionName + '" not supported on this browser.');
}
export function createVertexShader(gl, vertexShaderSource) {
    const vertexShader = throwIfNull(gl, () => gl.createShader(gl.VERTEX_SHADER), 'Unable to create vertex WebGLShader.');
    callAndCheck(gl, () => gl.shaderSource(vertexShader, vertexShaderSource));
    callAndCheck(gl, () => gl.compileShader(vertexShader));
    if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(vertexShader));
        throw new Error('Failed to compile vertex shader.');
    }
    return vertexShader;
}
export function createFragmentShader(gl, fragmentShaderSource) {
    const fragmentShader = throwIfNull(gl, () => gl.createShader(gl.FRAGMENT_SHADER), 'Unable to create fragment WebGLShader.');
    callAndCheck(gl, () => gl.shaderSource(fragmentShader, fragmentShaderSource));
    callAndCheck(gl, () => gl.compileShader(fragmentShader));
    if (env().get('ENGINE_COMPILE_ONLY')) {
        return fragmentShader;
    }
    if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
        logShaderSourceAndInfoLog(fragmentShaderSource, gl.getShaderInfoLog(fragmentShader));
        throw new Error('Failed to compile fragment shader.');
    }
    return fragmentShader;
}
const lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
export function logShaderSourceAndInfoLog(shaderSource, shaderInfoLog) {
    const lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
    if (lineNumberRegexResult == null) {
        console.log(`Couldn't parse line number in error: ${shaderInfoLog}`);
        console.log(shaderSource);
        return;
    }
    const lineNumber = +lineNumberRegexResult[1];
    const shaderLines = shaderSource.split('\n');
    const pad = shaderLines.length.toString().length + 2;
    const linesWithLineNumbers = shaderLines.map((line, lineNumber) => util.rightPad((lineNumber + 1).toString(), pad) + line);
    let maxLineLength = 0;
    for (let i = 0; i < linesWithLineNumbers.length; i++) {
        maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
    }
    const beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
    const errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
    const afterErrorLines = linesWithLineNumbers.slice(lineNumber);
    console.log(beforeErrorLines.join('\n'));
    console.log(shaderInfoLog.split('\n')[0]);
    console.log(`%c ${util.rightPad(errorLine[0], maxLineLength)}`, 'border:1px solid red; background-color:#e3d2d2; color:#a61717');
    console.log(afterErrorLines.join('\n'));
}
export function createProgram(gl) {
    return throwIfNull(gl, () => gl.createProgram(), 'Unable to create WebGLProgram.');
}
export function linkProgram(gl, program) {
    callAndCheck(gl, () => gl.linkProgram(program));
    if (env().get('ENGINE_COMPILE_ONLY')) {
        return;
    }
    if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Failed to link vertex and fragment shaders.');
    }
}
/// validateProgram is effectively "If we `useProgram(program); drawArrays();`,
/// give feedback in log about perf/correctness warnings or errors that would
/// occur."
/// So make sure we set up all vertex/texture/sampler/uniform data before
/// calling validateProgram!
export function validateProgram(gl, program) {
    callAndCheck(gl, () => gl.validateProgram(program));
    if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Shader program validation failed.');
    }
}
export function createStaticVertexBuffer(gl, data) {
    const buffer = throwIfNull(gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
    callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW));
    return buffer;
}
export function createStaticIndexBuffer(gl, data) {
    const buffer = throwIfNull(gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
    callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW));
    return buffer;
}
export function getNumChannels() {
    if (env().getNumber('WEBGL_VERSION') === 2) {
        return 1;
    }
    return 4;
}
export function createTexture(gl) {
    return throwIfNull(gl, () => gl.createTexture(), 'Unable to create WebGLTexture.');
}
export function validateTextureSize(width, height) {
    const maxTextureSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
    if ((width <= 0) || (height <= 0)) {
        const requested = `[${width}x${height}]`;
        throw new Error('Requested texture size ' + requested + ' is invalid.');
    }
    if ((width > maxTextureSize) || (height > maxTextureSize)) {
        const requested = `[${width}x${height}]`;
        const max = `[${maxTextureSize}x${maxTextureSize}]`;
        throw new Error('Requested texture size ' + requested +
            ' greater than WebGL maximum on this browser / GPU ' + max + '.');
    }
}
export function createFramebuffer(gl) {
    return throwIfNull(gl, () => gl.createFramebuffer(), 'Unable to create WebGLFramebuffer.');
}
export function bindVertexBufferToProgramAttribute(gl, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
    const loc = gl.getAttribLocation(program, attribute);
    if (loc === -1) {
        // The GPU compiler decided to strip out this attribute because it's unused,
        // thus no need to bind.
        return false;
    }
    callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes));
    callAndCheck(gl, () => gl.enableVertexAttribArray(loc));
    return true;
}
export function bindTextureUnit(gl, texture, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
    callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));
}
export function unbindTextureUnit(gl, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
    callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
}
export function getProgramUniformLocationOrThrow(gl, program, uniformName) {
    return throwIfNull(gl, () => gl.getUniformLocation(program, uniformName), 'uniform "' + uniformName + '" not present in program.');
}
export function getProgramUniformLocation(gl, program, uniformName) {
    return gl.getUniformLocation(program, uniformName);
}
export function bindTextureToProgramUniformSampler(gl, texture, uniformSamplerLocation, textureUnit) {
    callAndCheck(gl, () => bindTextureUnit(gl, texture, textureUnit));
    callAndCheck(gl, () => gl.uniform1i(uniformSamplerLocation, textureUnit));
}
export function bindCanvasToFramebuffer(gl) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
    callAndCheck(gl, () => gl.viewport(0, 0, gl.canvas.width, gl.canvas.height));
    callAndCheck(gl, () => gl.scissor(0, 0, gl.canvas.width, gl.canvas.height));
}
export function bindColorTextureToFramebuffer(gl, texture, framebuffer) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
    callAndCheck(gl, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0));
}
export function unbindColorTextureFromFramebuffer(gl, framebuffer) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
    callAndCheck(gl, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0));
}
export function validateFramebuffer(gl) {
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
    }
}
export function getFramebufferErrorMessage(gl, status) {
    switch (status) {
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
        case gl.FRAMEBUFFER_UNSUPPORTED:
            return 'FRAMEBUFFER_UNSUPPORTED';
        default:
            return `unknown error ${status}`;
    }
}
function throwIfNull(gl, returnTOrNull, failureMessage) {
    const tOrNull = callAndCheck(gl, () => returnTOrNull());
    if (tOrNull == null) {
        throw new Error(failureMessage);
    }
    return tOrNull;
}
function validateTextureUnit(gl, textureUnit) {
    const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
    const glTextureUnit = textureUnit + gl.TEXTURE0;
    if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
        const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
        throw new Error(`textureUnit must be in ${textureUnitRange}.`);
    }
}
export function getBatchDim(shape, dimsToSkip = 2) {
    return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
}
export function getRowsCols(shape) {
    if (shape.length === 0) {
        throw Error('Cannot get rows and columns of an empty shape array.');
    }
    return [
        shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
    ];
}
export function getShapeAs3D(shape) {
    let shapeAs3D = [1, 1, 1];
    const isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
    if (!isScalar) {
        shapeAs3D =
            [getBatchDim(shape), ...getRowsCols(shape)];
    }
    return shapeAs3D;
}
export function getTextureShapeFromLogicalShape(logShape, isPacked = false) {
    let maxTexSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
    let maxSizeForNarrowTex = env().getNumber('WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE');
    if (maxSizeForNarrowTex === Infinity &&
        env().getBool('WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE')) {
        maxSizeForNarrowTex = maxTexSize / 2;
    }
    if (isPacked) {
        maxTexSize = maxTexSize * 2;
        maxSizeForNarrowTex = maxSizeForNarrowTex * 2;
        // This logic ensures we accurately count the number of packed texels needed
        // to accommodate the tensor. We can only pack values in the same texel if
        // they are from adjacent pairs of rows/cols within the same batch. So if a
        // tensor has 3 rows, we pretend it has 4 rows in order to account for the
        // fact that the texels containing the third row are half empty.
        logShape = logShape.map((d, i) => i >= logShape.length - 2 ?
            util.nearestLargerEven(logShape[i]) :
            logShape[i]);
        // Packed texture height is at least 2 (the channel height of a single
        // texel).
        if (logShape.length === 1) {
            logShape = [2, logShape[0]];
        }
    }
    // If logical shape is 2, we don't squeeze, since we want to match physical.
    if (logShape.length !== 2) {
        const squeezeResult = util.squeezeShape(logShape);
        logShape = squeezeResult.newShape;
    }
    let size = util.sizeFromShape(logShape);
    let textureShape = null;
    if (logShape.length <= 1 && size <= maxTexSize) {
        textureShape = [1, size];
    }
    else if (logShape.length === 2 && logShape[0] <= maxTexSize &&
        logShape[1] <= maxTexSize) {
        textureShape = logShape;
    }
    else if (logShape.length === 3 && logShape[0] * logShape[1] <= maxTexSize &&
        logShape[2] <= maxTexSize) {
        textureShape = [logShape[0] * logShape[1], logShape[2]];
    }
    else if (logShape.length === 3 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] <= maxTexSize) {
        textureShape = [logShape[0], logShape[1] * logShape[2]];
    }
    else if (logShape.length === 4 &&
        logShape[0] * logShape[1] * logShape[2] <= maxTexSize &&
        logShape[3] <= maxTexSize) {
        textureShape = [logShape[0] * logShape[1] * logShape[2], logShape[3]];
    }
    else if (logShape.length === 4 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
        textureShape = [logShape[0], logShape[1] * logShape[2] * logShape[3]];
    }
    // true if one edge length is 1 (1 or 2, if packed), while another edge
    // length exceeds maxSizeForNarrowTex.
    const isLongNarrowTex = textureShape != null &&
        Math.max(...textureShape) > maxSizeForNarrowTex &&
        Math.min(...textureShape) <= (isPacked ? 2 : 1) &&
        Math.min(...textureShape) > 0;
    if (textureShape == null || isLongNarrowTex) {
        if (isPacked) {
            // For packed textures size equals the number of channels required to
            // accommodate the texture data. However in order to squarify such that
            // inner dimensions stay even, we rewrite size to equal the number of
            // texels. Then in the return statement we rehydrate the squarified
            // dimensions to channel units.
            const batchDim = getBatchDim(logShape);
            let rows = 2, cols = 2;
            if (logShape.length) {
                [rows, cols] = getRowsCols(logShape);
            }
            size = batchDim * (rows / 2) * (cols / 2);
            textureShape =
                util.sizeToSquarishShape(size).map(d => d * 2);
        }
        else {
            textureShape = util.sizeToSquarishShape(size);
        }
    }
    return textureShape;
}
function isEven(n) {
    return n % 2 === 0;
}
/**
 * This determines whether reshaping a packed texture requires rearranging
 * the data within the texture, assuming 2x2 packing.
 */
export function isReshapeFree(shape1, shape2) {
    shape1 = shape1.slice(-2);
    shape2 = shape2.slice(-2);
    if (util.arraysEqual(shape1, shape2)) {
        return true;
    }
    if (!shape1.length || !shape2.length) { // One of the shapes is a scalar.
        return true;
    }
    if (shape1[0] === 0 || shape1[1] === 0 || shape2[0] === 0 ||
        shape2[1] === 0) {
        return true;
    }
    if (shape1.length !== shape2.length) { // One of the shapes is a vector.
        const shape1Cols = shape1.slice(-1)[0];
        const shape2Cols = shape2.slice(-1)[0];
        if (shape1Cols === shape2Cols) {
            return true;
        }
        if (isEven(shape1Cols) && isEven(shape2Cols) &&
            (shape1[0] === 1 || shape2[0] === 1)) {
            return true;
        }
    }
    return shape1[1] === shape2[1] && isEven(shape1[0]) && isEven(shape2[0]);
}
// We cache webgl params because the environment gets reset between
// unit tests and we don't want to constantly query the WebGLContext for
// MAX_TEXTURE_SIZE.
let MAX_TEXTURE_SIZE;
let MAX_TEXTURES_IN_SHADER;
export function getWebGLMaxTextureSize(webGLVersion) {
    if (MAX_TEXTURE_SIZE == null) {
        const gl = getWebGLContext(webGLVersion);
        MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    }
    return MAX_TEXTURE_SIZE;
}
export function resetMaxTextureSize() {
    MAX_TEXTURE_SIZE = null;
}
export function resetMaxTexturesInShader() {
    MAX_TEXTURES_IN_SHADER = null;
}
export function getMaxTexturesInShader(webGLVersion) {
    if (MAX_TEXTURES_IN_SHADER == null) {
        const gl = getWebGLContext(webGLVersion);
        MAX_TEXTURES_IN_SHADER = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    }
    // We cap at 16 to avoid spurious runtime "memory exhausted" error.
    return Math.min(16, MAX_TEXTURES_IN_SHADER);
}
export function getWebGLDisjointQueryTimerVersion(webGLVersion) {
    if (webGLVersion === 0) {
        return 0;
    }
    let queryTimerVersion;
    const gl = getWebGLContext(webGLVersion);
    if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
        webGLVersion === 2) {
        queryTimerVersion = 2;
    }
    else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
        queryTimerVersion = 1;
    }
    else {
        queryTimerVersion = 0;
    }
    return queryTimerVersion;
}
export function hasExtension(gl, extensionName) {
    const ext = gl.getExtension(extensionName);
    return ext != null;
}
export function isWebGLVersionEnabled(webGLVersion) {
    try {
        const gl = getWebGLContext(webGLVersion);
        if (gl != null) {
            return true;
        }
    }
    catch (e) {
        console.log('Error when getting WebGL context: ', e);
        return false;
    }
    return false;
}
export function isCapableOfRenderingToFloatTexture(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
    }
    else {
        if (!hasExtension(gl, 'EXT_color_buffer_float')) {
            return false;
        }
    }
    const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
    return isFrameBufferComplete;
}
/**
 * Check if we can download values from a float/half-float texture.
 *
 * Note that for performance reasons we use binding a texture to a framebuffer
 * as a proxy for ability to download float values later using readPixels. The
 * texture params of this texture will not match those in readPixels exactly
 * but if we are unable to bind some kind of float texture to the frameBuffer
 * then we definitely will not be able to read float values from it.
 */
export function isDownloadFloatTextureEnabled(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
        if (!hasExtension(gl, 'WEBGL_color_buffer_float')) {
            return false;
        }
    }
    else {
        if (hasExtension(gl, 'EXT_color_buffer_float')) {
            return createFloatTextureAndBindToFramebuffer(gl);
        }
        const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
        if (hasExtension(gl, COLOR_BUFFER_HALF_FLOAT)) {
            const textureHalfFloatExtension = gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
            return createHalfFloatTextureAndBindToFramebuffer(gl, textureHalfFloatExtension);
        }
        return false;
    }
    const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
    return isFrameBufferComplete;
}
function createFloatTextureAndBindToFramebuffer(gl) {
    const texConfig = getTextureConfig(gl);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const width = 1;
    const height = 1;
    gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeFloat, null);
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isFrameBufferComplete;
}
function createHalfFloatTextureAndBindToFramebuffer(
// tslint:disable-next-line:no-any
gl, textureHalfFloatExtension) {
    const texConfig = getTextureConfig(gl, textureHalfFloatExtension);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const width = 1;
    const height = 1;
    gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatHalfFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeHalfFloat, null);
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isFrameBufferComplete;
}
export function isWebGLFenceEnabled(webGLVersion) {
    if (webGLVersion !== 2) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    // tslint:disable-next-line:no-any
    const isEnabled = gl.fenceSync != null;
    return isEnabled;
}
export function assertNotComplex(tensor, opName) {
    if (!Array.isArray(tensor)) {
        tensor = [tensor];
    }
    tensor.forEach(t => {
        if (t != null) {
            util.assert(t.dtype !== 'complex64', () => `${opName} does not support complex64 tensors ` +
                'in the WebGL backend.');
        }
    });
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2ViZ2xfdXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvd2ViZ2xfdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUFjLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDOUMsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRTVDLE1BQU0sVUFBVSxZQUFZLENBQUksRUFBeUIsRUFBRSxJQUFhO0lBQ3RFLE1BQU0sV0FBVyxHQUFHLElBQUksRUFBRSxDQUFDO0lBQzNCLElBQUksR0FBRyxFQUFFLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFO1FBQzFCLGVBQWUsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUNyQjtJQUNELE9BQU8sV0FBVyxDQUFDO0FBQ3JCLENBQUM7QUFFRCxTQUFTLGVBQWUsQ0FBQyxFQUF5QjtJQUNoRCxNQUFNLEtBQUssR0FBRyxFQUFFLENBQUMsUUFBUSxFQUFFLENBQUM7SUFDNUIsSUFBSSxLQUFLLEtBQUssRUFBRSxDQUFDLFFBQVEsRUFBRTtRQUN6QixNQUFNLElBQUksS0FBSyxDQUFDLGVBQWUsR0FBRyxvQkFBb0IsQ0FBQyxFQUFFLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQztLQUNwRTtBQUNILENBQUM7QUFFRCxxRUFBcUU7QUFDckUsTUFBTSxXQUFXLEdBQUcsT0FBTyxDQUFDO0FBQzVCLE1BQU0sV0FBVyxHQUFHLEtBQUssQ0FBQztBQUUxQixNQUFNLFVBQVUsZ0JBQWdCLENBQUMsR0FBVztJQUMxQyxJQUFJLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyw4QkFBOEIsQ0FBQyxJQUFJLEdBQUcsS0FBSyxDQUFDO1FBQzFELENBQUMsV0FBVyxHQUFH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Q0FBQyxJQUFJLFVBQVUsRUFBRTtRQUMzQyxZQUFZLEdBQUcsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ3pEO1NBQU0sSUFDSCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUM7UUFDckIsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVTtRQUNyRCxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVSxFQUFFO1FBQzdCLFlBQVksR0FBRyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ3ZFO1NBQU0sSUFDSCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVTtRQUNsRCxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVLEVBQUU7UUFDekQsWUFBWSxHQUFHLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDdkU7SUFFRCx1RUFBdUU7SUFDdkUsc0NBQXNDO0lBQ3RDLE1BQU0sZUFBZSxHQUFHLFlBQVksSUFBSSxJQUFJO1FBQ3hDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxZQUFZLENBQUMsR0FBRyxtQkFBbUI7UUFDL0MsSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLFlBQVksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxD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eXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2VudiwgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtnZXRXZWJHTENvbnRleHR9IGZyb20gJy4vY2FudmFzX3V0aWwnO1xuaW1wb3J0IHtnZXRUZXh0dXJlQ29uZmlnfSBmcm9tICcuL3RleF91dGlsJztcblxuZXhwb3J0IGZ1bmN0aW9uIGNhbGxBbmRDaGVjazxUPihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmdW5jOiAoKSA9PiBUKTogVCB7XG4gIGNvbnN0IHJldHVyblZhbHVlID0gZnVuYygpO1xuICBpZiAoZW52KCkuZ2V0Qm9vbCgnREVCVUcnKSkge1xuICAgIGNoZWNrV2ViR0xFcnJvcihnbCk7XG4gIH1cbiAgcmV0dXJuIHJldHVyblZhbHVlO1xufVxuXG5mdW5jdGlvbiBjaGVja1dlYkdMRXJyb3IoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjb25zdCBlcnJvciA9IGdsLmdldEVycm9yKCk7XG4gIGlmIChlcnJvciAhPT0gZ2wuTk9fRVJST1IpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1dlYkdMIEVycm9yOiAnICsgZ2V0V2ViR0xFcnJvck1lc3NhZ2UoZ2wsIGVycm9yKSk7XG4gIH1cbn1cblxuLy8gaHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvSGFsZi1wcmVjaXNpb25fZmxvYXRpbmctcG9pbnRfZm9ybWF0XG5jb25zdCBNSU5fRkxPQVQxNiA9IDUuOTZlLTg7XG5jb25zdCBNQVhfRkxPQVQxNiA9IDY1NTA0O1xuXG5leHBvcnQgZnVuY3Rpb24gY2FuQmVSZXByZXNlbnRlZChudW06IG51bWJlcik6IGJvb2xlYW4ge1xuICBpZiAoZW52KCkuZ2V0Qm9vbCgnV0VCR0xfUkVOREVSX0ZMT0FUMzJfRU5BQkxFRCcpIHx8IG51bSA9PT0gMCB8fFxuICAgICAgKE1JTl9GTE9BVDE2IDwgTWF0aC5hYnMobnVtKSAmJiBNYXRoLmFicyhudW0pIDwgTUFYX0ZMT0FUMTYpKSB7XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cbiAgcmV0dXJuIGZhbHNlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2ViR0xFcnJvck1lc3NhZ2UoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgc3RhdHVzOiBudW1iZXIpOiBzdHJpbmcge1xuICBzd2l0Y2ggKHN0YXR1cykge1xuICAgIGNhc2UgZ2wuTk9fRVJST1I6XG4gICAgICByZXR1cm4gJ05PX0VSUk9SJztcbiAgICBjYXNlIGdsLklOVkFMSURfRU5VTTpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9FTlVNJztcbiAgICBjYXNlIGdsLklOVkFMSURfVkFMVUU6XG4gICAgICByZXR1cm4gJ0lOVkFMSURfVkFMVUUnO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9PUEVSQVRJT046XG4gICAgICByZXR1cm4gJ0lOVkFMSURfT1BFUkFUSU9OJztcbiAgICBjYXNlIGdsLklOVkFMSURfRlJBTUVCVUZGRVJfT1BFUkFUSU9OOlxuICAgICAgcmV0dXJuICdJTlZBTElEX0ZSQU1FQlVGRkVSX09QRVJBVElPTic7XG4gICAgY2FzZSBnbC5PVVRfT0ZfTUVNT1JZOlxuICAgICAgcmV0dXJuICdPVVRfT0ZfTUVNT1JZJztcbiAgICBjYXNlIGdsLkNPTlRFWFRfTE9TVF9XRUJHTDpcbiAgICAgIHJldHVybiAnQ09OVEVYVF9MT1NUX1dFQkdMJztcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGBVbmtub3duIGVycm9yIGNvZGUgJHtzdGF0dXN9YDtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RXh0ZW5zaW9uT3JUaHJvdyhcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBleHRlbnNpb25OYW1lOiBzdHJpbmcpOiB7fSB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDx7fT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuZ2V0RXh0ZW5zaW9uKGV4dGVuc2lvbk5hbWUpLFxuICAgICAgJ0V4dGVuc2lvbiBcIicgKyBleHRlbnNpb25OYW1lICsgJ1wiIG5vdCBzdXBwb3J0ZWQgb24gdGhpcyBicm93c2VyLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVmVydGV4U2hhZGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHZlcnRleFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xTaGFkZXIge1xuICBjb25zdCB2ZXJ0ZXhTaGFkZXI6IFdlYkdMU2hhZGVyID0gdGhyb3dJZk51bGw8V2ViR0xTaGFkZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVNoYWRlcihnbC5WRVJURVhfU0hBREVSKSxcbiAgICAgICdVbmFibGUgdG8gY3JlYXRlIHZlcnRleCBXZWJHTFNoYWRlci4nKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zaGFkZXJTb3VyY2UodmVydGV4U2hhZGVyLCB2ZXJ0ZXhTaGFkZXJTb3VyY2UpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5jb21waWxlU2hhZGVyKHZlcnRleFNoYWRlcikpO1xuICBpZiAoZ2wuZ2V0U2hhZGVyUGFyYW1ldGVyKHZlcnRleFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFNoYWRlckluZm9Mb2codmVydGV4U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSB2ZXJ0ZXggc2hhZGVyLicpO1xuICB9XG4gIHJldHVybiB2ZXJ0ZXhTaGFkZXI7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVGcmFnbWVudFNoYWRlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmcmFnbWVudFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xTaGFkZXIge1xuICBjb25zdCBmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIgPSB0aHJvd0lmTnVsbDxXZWJHTFNoYWRlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlU2hhZGVyKGdsLkZSQUdNRU5UX1NIQURFUiksXG4gICAgICAnVW5hYmxlIHRvIGNyZWF0ZSBmcmFnbWVudCBXZWJHTFNoYWRlci4nKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zaGFkZXJTb3VyY2UoZnJhZ21lbnRTaGFkZXIsIGZyYWdtZW50U2hhZGVyU291cmNlKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY29tcGlsZVNoYWRlcihmcmFnbWVudFNoYWRlcikpO1xuICBpZiAoZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICByZXR1cm4gZnJhZ21lbnRTaGFkZXI7XG4gIH1cbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcihmcmFnbWVudFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGxvZ1NoYWRlclNvdXJjZUFuZEluZm9Mb2coXG4gICAgICAgIGZyYWdtZW50U2hhZGVyU291cmNlLCBnbC5nZXRTaGFkZXJJbmZvTG9nKGZyYWdtZW50U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSBmcmFnbWVudCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIGZyYWdtZW50U2hhZGVyO1xufVxuXG5jb25zdCBsaW5lTnVtYmVyUmVnZXggPSAvRVJST1I6IFswLTldKzooWzAtOV0rKTovZztcbmV4cG9ydCBmdW5jdGlvbiBsb2dTaGFkZXJTb3VyY2VBbmRJbmZvTG9nKFxuICAgIHNoYWRlclNvdXJjZTogc3RyaW5nLCBzaGFkZXJJbmZvTG9nOiBzdHJpbmcpIHtcbiAgY29uc3QgbGluZU51bWJlclJlZ2V4UmVzdWx0ID0gbGluZU51bWJlclJlZ2V4LmV4ZWMoc2hhZGVySW5mb0xvZyk7XG4gIGlmIChsaW5lTnVtYmVyUmVnZXhSZXN1bHQgPT0gbnVsbCkge1xuICAgIGNvbnNvbGUubG9nKGBDb3VsZG4ndCBwYXJzZSBsaW5lIG51bWJlciBpbiBlcnJvcjogJHtzaGFkZXJJbmZvTG9nfWApO1xuICAgIGNvbnNvbGUubG9nKHNoYWRlclNvdXJjZSk7XG4gICAgcmV0dXJuO1xuICB9XG5cbiAgY29uc3QgbGluZU51bWJlciA9ICtsaW5lTnVtYmVyUmVnZXhSZXN1bHRbMV07XG5cbiAgY29uc3Qgc2hhZGVyTGluZXMgPSBzaGFkZXJTb3VyY2Uuc3BsaXQoJ1xcbicpO1xuICBjb25zdCBwYWQgPSBzaGFkZXJMaW5lcy5sZW5ndGgudG9TdHJpbmcoKS5sZW5ndGggKyAyO1xuICBjb25zdCBsaW5lc1dpdGhMaW5lTnVtYmVycyA9IHNoYWRlckxpbmVzLm1hcChcbiAgICAgIChsaW5lLCBsaW5lTnVtYmVyKSA9PlxuICAgICAgICAgIHV0aWwucmlnaHRQYWQoKGxpbmVOdW1iZXIgKyAxKS50b1N0cmluZygpLCBwYWQpICsgbGluZSk7XG4gIGxldCBtYXhMaW5lTGVuZ3RoID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBsaW5lc1dpdGhMaW5lTnVtYmVycy5sZW5ndGg7IGkrKykge1xuICAgIG1heExpbmVMZW5ndGggPSBNYXRoLm1heChsaW5lc1dpdGhMaW5lTnVtYmVyc1tpXS5sZW5ndGgsIG1heExpbmVMZW5ndGgpO1xuICB9XG5cbiAgY29uc3QgYmVmb3JlRXJyb3JMaW5lcyA9IGxpbmVzV2l0aExpbmVOdW1iZXJzLnNsaWNlKDAsIGxpbmVOdW1iZXIgLSAxKTtcbiAgY29uc3QgZXJyb3JMaW5lID0gbGluZXNXaXRoTGluZU51bWJlcnMuc2xpY2UobGluZU51bWJlciAtIDEsIGxpbmVOdW1iZXIpO1xuICBjb25zdCBhZnRlckVycm9yTGluZXMgPSBsaW5lc1dpdGhMaW5lTnVtYmVycy5zbGljZShsaW5lTnVtYmVyKTtcblxuICBjb25zb2xlLmxvZyhiZWZvcmVFcnJvckxpbmVzLmpvaW4oJ1xcbicpKTtcbiAgY29uc29sZS5sb2coc2hhZGVySW5mb0xvZy5zcGxpdCgnXFxuJylbMF0pO1xuICBjb25zb2xlLmxvZyhcbiAgICAgIGAlYyAke3V0aWwucmlnaHRQYWQoZXJyb3JMaW5lWzBdLCBtYXhMaW5lTGVuZ3RoKX1gLFxuICAgICAgJ2JvcmRlcjoxcHggc29saWQgcmVkOyBiYWNrZ3JvdW5kLWNvbG9yOiNlM2QyZDI7IGNvbG9yOiNhNjE3MTcnKTtcbiAgY29uc29sZS5sb2coYWZ0ZXJFcnJvckxpbmVzLmpvaW4oJ1xcbicpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVByb2dyYW0oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMUHJvZ3JhbSB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFByb2dyYW0+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVByb2dyYW0oKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xQcm9ncmFtLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbGlua1Byb2dyYW0oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wubGlua1Byb2dyYW0ocHJvZ3JhbSkpO1xuICBpZiAoZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICByZXR1cm47XG4gIH1cbiAgaWYgKGdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBsaW5rIHZlcnRleCBhbmQgZnJhZ21lbnQgc2hhZGVycy4nKTtcbiAgfVxufVxuXG4vLy8gdmFsaWRhdGVQcm9ncmFtIGlzIGVmZmVjdGl2ZWx5IFwiSWYgd2UgYHVzZVByb2dyYW0ocHJvZ3JhbSk7IGRyYXdBcnJheXMoKTtgLFxuLy8vIGdpdmUgZmVlZGJhY2sgaW4gbG9nIGFib3V0IHBlcmYvY29ycmVjdG5lc3Mgd2FybmluZ3Mgb3IgZXJyb3JzIHRoYXQgd291bGRcbi8vLyBvY2N1ci5cIlxuLy8vIFNvIG1ha2Ugc3VyZSB3ZSBzZXQgdXAgYWxsIHZlcnRleC90ZXh0dXJlL3NhbXBsZXIvdW5pZm9ybSBkYXRhIGJlZm9yZVxuLy8vIGNhbGxpbmcgdmFsaWRhdGVQcm9ncmFtIVxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlUHJvZ3JhbShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52YWxpZGF0ZVByb2dyYW0ocHJvZ3JhbSkpO1xuICBpZiAoZ2wuZ2V0UHJvZ3JhbVBhcmFtZXRlcihwcm9ncmFtLCBnbC5WQUxJREFURV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1NoYWRlciBwcm9ncmFtIHZhbGlkYXRpb24gZmFpbGVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTdGF0aWNWZXJ0ZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogRmxvYXQzMkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkFSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGRhdGE6IFVpbnQxNkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCwgKCkgPT4gZ2wuYnVmZmVyRGF0YShnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE51bUNoYW5uZWxzKCk6IG51bWJlciB7XG4gIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSA9PT0gMikge1xuICAgIHJldHVybiAxO1xuICB9XG4gIHJldHVybiA0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVGV4dHVyZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xUZXh0dXJlIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMVGV4dHVyZT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlVGV4dHVyZSgpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTFRleHR1cmUuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZVRleHR1cmVTaXplKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVTaXplID0gZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9NQVhfVEVYVFVSRV9TSVpFJyk7XG4gIGlmICgod2lkdGggPD0gMCkgfHwgKGhlaWdodCA8PSAwKSkge1xuICAgIGNvbnN0IHJlcXVlc3RlZCA9IGBbJHt3aWR0aH14JHtoZWlnaHR9XWA7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgKyAnIGlzIGludmFsaWQuJyk7XG4gIH1cbiAgaWYgKCh3aWR0aCA+IG1heFRleHR1cmVTaXplKSB8fCAoaGVpZ2h0ID4gbWF4VGV4dHVyZVNpemUpKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gYFske3dpZHRofXgke2hlaWdodH1dYDtcbiAgICBjb25zdCBtYXggPSBgWyR7bWF4VGV4dHVyZVNpemV9eCR7bWF4VGV4dHVyZVNpemV9XWA7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnUmVxdWVzdGVkIHRleHR1cmUgc2l6ZSAnICsgcmVxdWVzdGVkICtcbiAgICAgICAgJyBncmVhdGVyIHRoYW4gV2ViR0wgbWF4aW11bSBvbiB0aGlzIGJyb3dzZXIgLyBHUFUgJyArIG1heCArICcuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTEZyYW1lYnVmZmVyIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMRnJhbWVidWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUZyYW1lYnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMRnJhbWVidWZmZXIuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVmVydGV4QnVmZmVyVG9Qcm9ncmFtQXR0cmlidXRlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYXR0cmlidXRlOiBzdHJpbmcsXG4gICAgYnVmZmVyOiBXZWJHTEJ1ZmZlciwgYXJyYXlFbnRyaWVzUGVySXRlbTogbnVtYmVyLCBpdGVtU3RyaWRlSW5CeXRlczogbnVtYmVyLFxuICAgIGl0ZW1PZmZzZXRJbkJ5dGVzOiBudW1iZXIpOiBib29sZWFuIHtcbiAgY29uc3QgbG9jID0gZ2wuZ2V0QXR0cmliTG9jYXRpb24ocHJvZ3JhbSwgYXR0cmlidXRlKTtcbiAgaWYgKGxvYyA9PT0gLTEpIHtcbiAgICAvLyBUaGUgR1BVIGNvbXBpbGVyIGRlY2lkZWQgdG8gc3RyaXAgb3V0IHRoaXMgYXR0cmlidXRlIGJlY2F1c2UgaXQncyB1bnVzZWQsXG4gICAgLy8gdGh1cyBubyBuZWVkIHRvIGJpbmQuXG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnZlcnRleEF0dHJpYlBvaW50ZXIoXG4gICAgICAgICAgbG9jLCBhcnJheUVudHJpZXNQZXJJdGVtLCBnbC5GTE9BVCwgZmFsc2UsIGl0ZW1TdHJpZGVJbkJ5dGVzLFxuICAgICAgICAgIGl0ZW1PZmZzZXRJbkJ5dGVzKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlVmVydGV4QXR0cmliQXJyYXkobG9jKSk7XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFRleHR1cmVVbml0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5iaW5kVGV4dHVyZVVuaXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLFxuICAgIHVuaWZvcm1OYW1lOiBzdHJpbmcpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFVuaWZvcm1Mb2NhdGlvbj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIHVuaWZvcm1OYW1lKSxcbiAgICAgICd1bmlmb3JtIFwiJyArIHVuaWZvcm1OYW1lICsgJ1wiIG5vdCBwcmVzZW50IGluIHByb2dyYW0uJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSxcbiAgICB1bmlmb3JtTmFtZTogc3RyaW5nKTogV2ViR0xVbmlmb3JtTG9jYXRpb24ge1xuICByZXR1cm4gZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIHVuaWZvcm1OYW1lKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRUZXh0dXJlVG9Qcm9ncmFtVW5pZm9ybVNhbXBsZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHVuaWZvcm1TYW1wbGVyTG9jYXRpb246IFdlYkdMVW5pZm9ybUxvY2F0aW9uLCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gYmluZFRleHR1cmVVbml0KGdsLCB0ZXh0dXJlLCB0ZXh0dXJlVW5pdCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnVuaWZvcm0xaSh1bmlmb3JtU2FtcGxlckxvY2F0aW9uLCB0ZXh0dXJlVW5pdCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZENhbnZhc1RvRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZpZXdwb3J0KDAsIDAsIGdsLmNhbnZhcy53aWR0aCwgZ2wuY2FudmFzLmhlaWdodCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNjaXNzb3IoMCwgMCwgZ2wuY2FudmFzLndpZHRoLCBnbC5jYW52YXMuaGVpZ2h0KSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICAgICAgZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCB0ZXh0dXJlLCAwKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICAgICAgZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCBudWxsLCAwKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZUZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgY29uc3Qgc3RhdHVzID0gZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUik7XG4gIGlmIChzdGF0dXMgIT09IGdsLkZSQU1FQlVGRkVSX0NPTVBMRVRFKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnRXJyb3IgYmluZGluZyBmcmFtZWJ1ZmZlcjogJyArIGdldEZyYW1lYnVmZmVyRXJyb3JNZXNzYWdlKGdsLCBzdGF0dXMpKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhbWVidWZmZXJFcnJvck1lc3NhZ2UoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgc3RhdHVzOiBudW1iZXIpOiBzdHJpbmcge1xuICBzd2l0Y2ggKHN0YXR1cykge1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9BVFRBQ0hNRU5UOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0FUVEFDSE1FTlQnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfTUlTU0lOR19BVFRBQ0hNRU5UJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfRElNRU5TSU9OUzpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9ESU1FTlNJT05TJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9VTlNVUFBPUlRFRCc7XG4gICAgZGVmYXVsdDpcbiAgICAgIHJldHVybiBgdW5rbm93biBlcnJvciAke3N0YXR1c31gO1xuICB9XG59XG5cbmZ1bmN0aW9uIHRocm93SWZOdWxsPFQ+KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJldHVyblRPck51bGw6ICgpID0+IFQgfCBudWxsLFxuICAgIGZhaWx1cmVNZXNzYWdlOiBzdHJpbmcpOiBUIHtcbiAgY29uc3QgdE9yTnVsbDogVHxudWxsID0gY2FsbEFuZENoZWNrKGdsLCAoKSA9PiByZXR1cm5UT3JOdWxsKCkpO1xuICBpZiAodE9yTnVsbCA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGZhaWx1cmVNZXNzYWdlKTtcbiAgfVxuICByZXR1cm4gdE9yTnVsbDtcbn1cblxuZnVuY3Rpb24gdmFsaWRhdGVUZXh0dXJlVW5pdChnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVVbml0ID0gZ2wuTUFYX0NPTUJJTkVEX1RFWFRVUkVfSU1BR0VfVU5JVFMgLSAxO1xuICBjb25zdCBnbFRleHR1cmVVbml0ID0gdGV4dHVyZVVuaXQgKyBnbC5URVhUVVJFMDtcbiAgaWYgKGdsVGV4dHVyZVVuaXQgPCBnbC5URVhUVVJFMCB8fCBnbFRleHR1cmVVbml0ID4gbWF4VGV4dHVyZVVuaXQpIHtcbiAgICBjb25zdCB0ZXh0dXJlVW5pdFJhbmdlID0gYFtnbC5URVhUVVJFMCwgZ2wuVEVYVFVSRSR7bWF4VGV4dHVyZVVuaXR9XWA7XG4gICAgdGhyb3cgbmV3IEVycm9yKGB0ZXh0dXJlVW5pdCBtdXN0IGJlIGluICR7dGV4dHVyZVVuaXRSYW5nZX0uYCk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEJhdGNoRGltKHNoYXBlOiBudW1iZXJbXSwgZGltc1RvU2tpcCA9IDIpOiBudW1iZXIge1xuICByZXR1cm4gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlLnNsaWNlKDAsIHNoYXBlLmxlbmd0aCAtIGRpbXNUb1NraXApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFJvd3NDb2xzKHNoYXBlOiBudW1iZXJbXSk6IFtudW1iZXIsIG51bWJlcl0ge1xuICBpZiAoc2hhcGUubGVuZ3RoID09PSAwKSB7XG4gICAgdGhyb3cgRXJyb3IoJ0Nhbm5vdCBnZXQgcm93cyBhbmQgY29sdW1ucyBvZiBhbiBlbXB0eSBzaGFwZSBhcnJheS4nKTtcbiAgfVxuXG4gIHJldHVybiBbXG4gICAgc2hhcGUubGVuZ3RoID4gMSA/IHNoYXBlW3NoYXBlLmxlbmd0aCAtIDJdIDogMSwgc2hhcGVbc2hhcGUubGVuZ3RoIC0gMV1cbiAgXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFNoYXBlQXMzRChzaGFwZTogbnVtYmVyW10pOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICBsZXQgc2hhcGVBczNEOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbMSwgMSwgMV07XG4gIGNvbnN0IGlzU2NhbGFyID0gc2hhcGUubGVuZ3RoID09PSAwIHx8IChzaGFwZS5sZW5ndGggPT09IDEgJiYgc2hhcGVbMF0gPT09IDEpO1xuICBpZiAoIWlzU2NhbGFyKSB7XG4gICAgc2hhcGVBczNEID1cbiAgICAgICAgW2dldEJhdGNoRGltKHNoYXBlKSwgLi4uZ2V0Um93c0NvbHMoc2hhcGUpXSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIH1cbiAgcmV0dXJuIHNoYXBlQXMzRDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFRleHR1cmVTaGFwZUZyb21Mb2dpY2FsU2hhcGUoXG4gICAgbG9nU2hhcGU6IG51bWJlcltdLCBpc1BhY2tlZCA9IGZhbHNlKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGxldCBtYXhUZXhTaXplID0gZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9NQVhfVEVYVFVSRV9TSVpFJyk7XG4gIGxldCBtYXhTaXplRm9yTmFycm93VGV4ID1cbiAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfTUFYX1NJWkVfRk9SX05BUlJPV19URVhUVVJFJyk7XG4gIGlmIChtYXhTaXplRm9yTmFycm93VGV4ID09PSBJbmZpbml0eSAmJlxuICAgICAgZW52KCkuZ2V0Qm9vbCgnV0VCR0xfQVVUT19TUVVBUklGWV9OQVJST1dfVEVYVFVSRV9TSEFQRScpKSB7XG4gICAgbWF4U2l6ZUZvck5hcnJvd1RleCA9IG1heFRleFNpemUgLyAyO1xuICB9XG5cbiAgaWYgKGlzUGFja2VkKSB7XG4gICAgbWF4VGV4U2l6ZSA9IG1heFRleFNpemUgKiAyO1xuICAgIG1heFNpemVGb3JOYXJyb3dUZXggPSBtYXhTaXplRm9yTmFycm93VGV4ICogMjtcblxuICAgIC8vIFRoaXMgbG9naWMgZW5zdXJlcyB3ZSBhY2N1cmF0ZWx5IGNvdW50IHRoZSBudW1iZXIgb2YgcGFja2VkIHRleGVscyBuZWVkZWRcbiAgICAvLyB0byBhY2NvbW1vZGF0ZSB0aGUgdGVuc29yLiBXZSBjYW4gb25seSBwYWNrIHZhbHVlcyBpbiB0aGUgc2FtZSB0ZXhlbCBpZlxuICAgIC8vIHRoZXkgYXJlIGZyb20gYWRqYWNlbnQgcGFpcnMgb2Ygcm93cy9jb2xzIHdpdGhpbiB0aGUgc2FtZSBiYXRjaC4gU28gaWYgYVxuICAgIC8vIHRlbnNvciBoYXMgMyByb3dzLCB3ZSBwcmV0ZW5kIGl0IGhhcyA0IHJvd3MgaW4gb3JkZXIgdG8gYWNjb3VudCBmb3IgdGhlXG4gICAgLy8gZmFjdCB0aGF0IHRoZSB0ZXhlbHMgY29udGFpbmluZyB0aGUgdGhpcmQgcm93IGFyZSBoYWxmIGVtcHR5LlxuICAgIGxvZ1NoYXBlID0gbG9nU2hhcGUubWFwKFxuICAgICAgICAoZCwgaSkgPT4gaSA+PSBsb2dTaGFwZS5sZW5ndGggLSAyID9cbiAgICAgICAgICAgIHV0aWwubmVhcmVzdExhcmdlckV2ZW4obG9nU2hhcGVbaV0pIDpcbiAgICAgICAgICAgIGxvZ1NoYXBlW2ldKTtcblxuICAgIC8vIFBhY2tlZCB0ZXh0dXJlIGhlaWdodCBpcyBhdCBsZWFzdCAyICh0aGUgY2hhbm5lbCBoZWlnaHQgb2YgYSBzaW5nbGVcbiAgICAvLyB0ZXhlbCkuXG4gICAgaWYgKGxvZ1NoYXBlLmxlbmd0aCA9PT0gMSkge1xuICAgICAgbG9nU2hhcGUgPSBbMiwgbG9nU2hhcGVbMF1dO1xuICAgIH1cbiAgfVxuXG4gIC8vIElmIGxvZ2ljYWwgc2hhcGUgaXMgMiwgd2UgZG9uJ3Qgc3F1ZWV6ZSwgc2luY2Ugd2Ugd2FudCB0byBtYXRjaCBwaHlzaWNhbC5cbiAgaWYgKGxvZ1NoYXBlLmxlbmd0aCAhPT0gMikge1xuICAgIGNvbnN0IHNxdWVlemVSZXN1bHQgPSB1dGlsLnNxdWVlemVTaGFwZShsb2dTaGFwZSk7XG4gICAgbG9nU2hhcGUgPSBzcXVlZXplUmVzdWx0Lm5ld1NoYXBlO1xuICB9XG5cbiAgbGV0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUobG9nU2hhcGUpO1xuICBsZXQgdGV4dHVyZVNoYXBlOiBbbnVtYmVyLCBudW1iZXJdID0gbnVsbDtcbiAgaWYgKGxvZ1NoYXBlLmxlbmd0aCA8PSAxICYmIHNpemUgPD0gbWF4VGV4U2l6ZSkge1xuICAgIHRleHR1cmVTaGFwZSA9IFsxLCBzaXplXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gMiAmJiBsb2dTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dTaGFwZVsxXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgdGV4dHVyZVNoYXBlID0gbG9nU2hhcGUgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gMyAmJiBsb2dTaGFwZVswXSAqIGxvZ1NoYXBlWzFdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzJdIDw9IG1heFRleFNpemUpIHtcbiAgICB0ZXh0dXJlU2hhcGUgPSBbbG9nU2hhcGVbMF0gKiBsb2dTaGFwZVsxXSwgbG9nU2hhcGVbMl1dO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSAzICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl0gPD0gbWF4VGV4U2l6ZSkge1xuICAgIHRleHR1cmVTaGFwZSA9IFtsb2dTaGFwZVswXSwgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXV07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dTaGFwZS5sZW5ndGggPT09IDQgJiZcbiAgICAgIGxvZ1NoYXBlWzBdICogbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dTaGFwZVszXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgdGV4dHVyZVNoYXBlID0gW2xvZ1NoYXBlWzBdICogbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSwgbG9nU2hhcGVbM11dO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSA0ICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl0gKiBsb2dTaGFwZVszXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgdGV4dHVyZVNoYXBlID0gW2xvZ1NoYXBlWzBdLCBsb2dTaGFwZVsxXSAqIGxvZ1NoYXBlWzJdICogbG9nU2hhcGVbM11dO1xuICB9XG5cbiAgLy8gdHJ1ZSBpZiBvbmUgZWRnZSBsZW5ndGggaXMgMSAoMSBvciAyLCBpZiBwYWNrZWQpLCB3aGlsZSBhbm90aGVyIGVkZ2VcbiAgLy8gbGVuZ3RoIGV4Y2VlZHMgbWF4U2l6ZUZvck5hcnJvd1RleC5cbiAgY29uc3QgaXNMb25nTmFycm93VGV4ID0gdGV4dHVyZVNoYXBlICE9IG51bGwgJiZcbiAgICAgIE1hdGgubWF4KC4uLnRleHR1cmVTaGFwZSkgPiBtYXhTaXplRm9yTmFycm93VGV4ICYmXG4gICAgICBNYXRoLm1pbiguLi50ZXh0dXJlU2hhcGUpIDw9IChpc1BhY2tlZCA/IDIgOiAxKSAmJlxuICAgICAgTWF0aC5taW4oLi4udGV4dHVyZVNoYXBlKSA+IDA7XG5cbiAgaWYgKHRleHR1cmVTaGFwZSA9PSBudWxsIHx8IGlzTG9uZ05hcnJvd1RleCkge1xuICAgIGlmIChpc1BhY2tlZCkge1xuICAgICAgLy8gRm9yIHBhY2tlZCB0ZXh0dXJlcyBzaXplIGVxdWFscyB0aGUgbnVtYmVyIG9mIGNoYW5uZWxzIHJlcXVpcmVkIHRvXG4gICAgICAvLyBhY2NvbW1vZGF0ZSB0aGUgdGV4dHVyZSBkYXRhLiBIb3dldmVyIGluIG9yZGVyIHRvIHNxdWFyaWZ5IHN1Y2ggdGhhdFxuICAgICAgLy8gaW5uZXIgZGltZW5zaW9ucyBzdGF5IGV2ZW4sIHdlIHJld3JpdGUgc2l6ZSB0byBlcXVhbCB0aGUgbnVtYmVyIG9mXG4gICAgICAvLyB0ZXhlbHMuIFRoZW4gaW4gdGhlIHJldHVybiBzdGF0ZW1lbnQgd2UgcmVoeWRyYXRlIHRoZSBzcXVhcmlmaWVkXG4gICAgICAvLyBkaW1lbnNpb25zIHRvIGNoYW5uZWwgdW5pdHMuXG5cbiAgICAgIGNvbnN0IGJhdGNoRGltID0gZ2V0QmF0Y2hEaW0obG9nU2hhcGUpO1xuICAgICAgbGV0IHJvd3MgPSAyLCBjb2xzID0gMjtcbiAgICAgIGlmIChsb2dTaGFwZS5sZW5ndGgpIHtcbiAgICAgICAgW3Jvd3MsIGNvbHNdID0gZ2V0Um93c0NvbHMobG9nU2hhcGUpO1xuICAgICAgfVxuICAgICAgc2l6ZSA9IGJhdGNoRGltICogKHJvd3MgLyAyKSAqIChjb2xzIC8gMik7XG4gICAgICB0ZXh0dXJlU2hhcGUgPVxuICAgICAgICAgIHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplKS5tYXAoZCA9PiBkICogMikgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGV4dHVyZVNoYXBlID0gdXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKHNpemUpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiB0ZXh0dXJlU2hhcGU7XG59XG5cbmZ1bmN0aW9uIGlzRXZlbihuOiBudW1iZXIpOiBib29sZWFuIHtcbiAgcmV0dXJuIG4gJSAyID09PSAwO1xufVxuXG4vKipcbiAqIFRoaXMgZGV0ZXJtaW5lcyB3aGV0aGVyIHJlc2hhcGluZyBhIHBhY2tlZCB0ZXh0dXJlIHJlcXVpcmVzIHJlYXJyYW5naW5nXG4gKiB0aGUgZGF0YSB3aXRoaW4gdGhlIHRleHR1cmUsIGFzc3VtaW5nIDJ4MiBwYWNraW5nLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaXNSZXNoYXBlRnJlZShzaGFwZTE6IG51bWJlcltdLCBzaGFwZTI6IG51bWJlcltdKTogYm9vbGVhbiB7XG4gIHNoYXBlMSA9IHNoYXBlMS5zbGljZSgtMik7XG4gIHNoYXBlMiA9IHNoYXBlMi5zbGljZSgtMik7XG5cbiAgaWYgKHV0aWwuYXJyYXlzRXF1YWwoc2hhcGUxLCBzaGFwZTIpKSB7XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICBpZiAoIXNoYXBlMS5sZW5ndGggfHwgIXNoYXBlMi5sZW5ndGgpIHsgIC8vIE9uZSBvZiB0aGUgc2hhcGVzIGlzIGEgc2NhbGFyLlxuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgaWYgKHNoYXBlMVswXSA9PT0gMCB8fCBzaGFwZTFbMV0gPT09IDAgfHwgc2hhcGUyWzBdID09PSAwIHx8XG4gICAgICBzaGFwZTJbMV0gPT09IDApIHtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGlmIChzaGFwZTEubGVuZ3RoICE9PSBzaGFwZTIubGVuZ3RoKSB7ICAvLyBPbmUgb2YgdGhlIHNoYXBlcyBpcyBhIHZlY3Rvci5cbiAgICBjb25zdCBzaGFwZTFDb2xzID0gc2hhcGUxLnNsaWNlKC0xKVswXTtcbiAgICBjb25zdCBzaGFwZTJDb2xzID0gc2hhcGUyLnNsaWNlKC0xKVswXTtcbiAgICBpZiAoc2hhcGUxQ29scyA9PT0gc2hhcGUyQ29scykge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgaWYgKGlzRXZlbihzaGFwZTFDb2xzKSAmJiBpc0V2ZW4oc2hhcGUyQ29scykgJiZcbiAgICAgICAgKHNoYXBlMVswXSA9PT0gMSB8fCBzaGFwZTJbMF0gPT09IDEpKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHNoYXBlMVsxXSA9PT0gc2hhcGUyWzFdICYmIGlzRXZlbihzaGFwZTFbMF0pICYmIGlzRXZlbihzaGFwZTJbMF0pO1xufVxuXG4vLyBXZSBjYWNoZSB3ZWJnbCBwYXJhbXMgYmVjYXVzZSB0aGUgZW52aXJvbm1lbnQgZ2V0cyByZXNldCBiZXR3ZWVuXG4vLyB1bml0IHRlc3RzIGFuZCB3ZSBkb24ndCB3YW50IHRvIGNvbnN0YW50bHkgcXVlcnkgdGhlIFdlYkdMQ29udGV4dCBmb3Jcbi8vIE1BWF9URVhUVVJFX1NJWkUuXG5sZXQgTUFYX1RFWFRVUkVfU0laRTogbnVtYmVyO1xubGV0IE1BWF9URVhUVVJFU19JTl9TSEFERVI6IG51bWJlcjtcblxuZXhwb3J0IGZ1bmN0aW9uIGdldFdlYkdMTWF4VGV4dHVyZVNpemUod2ViR0xWZXJzaW9uOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAoTUFYX1RFWFRVUkVfU0laRSA9PSBudWxsKSB7XG4gICAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcbiAgICBNQVhfVEVYVFVSRV9TSVpFID0gZ2wuZ2V0UGFyYW1ldGVyKGdsLk1BWF9URVhUVVJFX1NJWkUpO1xuICB9XG4gIHJldHVybiBNQVhfVEVYVFVSRV9TSVpFO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcmVzZXRNYXhUZXh0dXJlU2l6ZSgpIHtcbiAgTUFYX1RFWFRVUkVfU0laRSA9IG51bGw7XG59XG5leHBvcnQgZnVuY3Rpb24gcmVzZXRNYXhUZXh0dXJlc0luU2hhZGVyKCkge1xuICBNQVhfVEVYVFVSRVNfSU5fU0hBREVSID0gbnVsbDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE1heFRleHR1cmVzSW5TaGFkZXIod2ViR0xWZXJzaW9uOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAoTUFYX1RFWFRVUkVTX0lOX1NIQURFUiA9PSBudWxsKSB7XG4gICAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcbiAgICBNQVhfVEVYVFVSRVNfSU5fU0hBREVSID0gZ2wuZ2V0UGFyYW1ldGVyKGdsLk1BWF9URVhUVVJFX0lNQUdFX1VOSVRTKTtcbiAgfVxuICAvLyBXZSBjYXAgYXQgMTYgdG8gYXZvaWQgc3B1cmlvdXMgcnVudGltZSBcIm1lbW9yeSBleGhhdXN0ZWRcIiBlcnJvci5cbiAgcmV0dXJuIE1hdGgubWluKDE2LCBNQVhfVEVYVFVSRVNfSU5fU0hBREVSKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFdlYkdMRGlzam9pbnRRdWVyeVRpbWVyVmVyc2lvbih3ZWJHTFZlcnNpb246IG51bWJlcik6XG4gICAgbnVtYmVyIHtcbiAgaWYgKHdlYkdMVmVyc2lvbiA9PT0gMCkge1xuICAgIHJldHVybiAwO1xuICB9XG5cbiAgbGV0IHF1ZXJ5VGltZXJWZXJzaW9uOiBudW1iZXI7XG4gIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KHdlYkdMVmVyc2lvbik7XG5cbiAgaWYgKGhhc0V4dGVuc2lvbihnbCwgJ0VYVF9kaXNqb2ludF90aW1lcl9xdWVyeV93ZWJnbDInKSAmJlxuICAgICAgd2ViR0xWZXJzaW9uID09PSAyKSB7XG4gICAgcXVlcnlUaW1lclZlcnNpb24gPSAyO1xuICB9IGVsc2UgaWYgKGhhc0V4dGVuc2lvbihnbCwgJ0VYVF9kaXNqb2ludF90aW1lcl9xdWVyeScpKSB7XG4gICAgcXVlcnlUaW1lclZlcnNpb24gPSAxO1xuICB9IGVsc2Uge1xuICAgIHF1ZXJ5VGltZXJWZXJzaW9uID0gMDtcbiAgfVxuICByZXR1cm4gcXVlcnlUaW1lclZlcnNpb247XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBoYXNFeHRlbnNpb24oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZXh0ZW5zaW9uTmFtZTogc3RyaW5nKSB7XG4gIGNvbnN0IGV4dCA9IGdsLmdldEV4dGVuc2lvbihleHRlbnNpb25OYW1lKTtcbiAgcmV0dXJuIGV4dCAhPSBudWxsO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNXZWJHTFZlcnNpb25FbmFibGVkKHdlYkdMVmVyc2lvbjogMXwyKSB7XG4gIHRyeSB7XG4gICAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcbiAgICBpZiAoZ2wgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICB9IGNhdGNoIChlKSB7XG4gICAgY29uc29sZS5sb2coJ0Vycm9yIHdoZW4gZ2V0dGluZyBXZWJHTCBjb250ZXh0OiAnLCBlKTtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgcmV0dXJuIGZhbHNlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNDYXBhYmxlT2ZSZW5kZXJpbmdUb0Zsb2F0VGV4dHVyZSh3ZWJHTFZlcnNpb246IG51bWJlcik6XG4gICAgYm9vbGVhbiB7XG4gIGlmICh3ZWJHTFZlcnNpb24gPT09IDApIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBjb25zdCBnbCA9IGdldFdlYkdMQ29udGV4dCh3ZWJHTFZlcnNpb24pO1xuXG4gIGlmICh3ZWJHTFZlcnNpb24gPT09IDEpIHtcbiAgICBpZiAoIWhhc0V4dGVuc2lvbihnbCwgJ09FU190ZXh0dXJlX2Zsb2F0JykpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgaWYgKCFoYXNFeHRlbnNpb24oZ2wsICdFWFRfY29sb3JfYnVmZmVyX2Zsb2F0JykpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH1cblxuICBjb25zdCBpc0ZyYW1lQnVmZmVyQ29tcGxldGUgPSBjcmVhdGVGbG9hdFRleHR1cmVBbmRCaW5kVG9GcmFtZWJ1ZmZlcihnbCk7XG4gIHJldHVybiBpc0ZyYW1lQnVmZmVyQ29tcGxldGU7XG59XG5cbi8qKlxuICogQ2hlY2sgaWYgd2UgY2FuIGRvd25sb2FkIHZhbHVlcyBmcm9tIGEgZmxvYXQvaGFsZi1mbG9hdCB0ZXh0dXJlLlxuICpcbiAqIE5vdGUgdGhhdCBmb3IgcGVyZm9ybWFuY2UgcmVhc29ucyB3ZSB1c2UgYmluZGluZyBhIHRleHR1cmUgdG8gYSBmcmFtZWJ1ZmZlclxuICogYXMgYSBwcm94eSBmb3IgYWJpbGl0eSB0byBkb3dubG9hZCBmbG9hdCB2YWx1ZXMgbGF0ZXIgdXNpbmcgcmVhZFBpeGVscy4gVGhlXG4gKiB0ZXh0dXJlIHBhcmFtcyBvZiB0aGlzIHRleHR1cmUgd2lsbCBub3QgbWF0Y2ggdGhvc2UgaW4gcmVhZFBpeGVscyBleGFjdGx5XG4gKiBidXQgaWYgd2UgYXJlIHVuYWJsZSB0byBiaW5kIHNvbWUga2luZCBvZiBmbG9hdCB0ZXh0dXJlIHRvIHRoZSBmcmFtZUJ1ZmZlclxuICogdGhlbiB3ZSBkZWZpbml0ZWx5IHdpbGwgbm90IGJlIGFibGUgdG8gcmVhZCBmbG9hdCB2YWx1ZXMgZnJvbSBpdC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzRG93bmxvYWRGbG9hdFRleHR1cmVFbmFibGVkKHdlYkdMVmVyc2lvbjogbnVtYmVyKTogYm9vbGVhbiB7XG4gIGlmICh3ZWJHTFZlcnNpb24gPT09IDApIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBjb25zdCBnbCA9IGdldFdlYkdMQ29udGV4dCh3ZWJHTFZlcnNpb24pO1xuXG4gIGlmICh3ZWJHTFZlcnNpb24gPT09IDEpIHtcbiAgICBpZiAoIWhhc0V4dGVuc2lvbihnbCwgJ09FU190ZXh0dXJlX2Zsb2F0JykpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgaWYgKCFoYXNFeHRlbnNpb24oZ2wsICdXRUJHTF9jb2xvcl9idWZmZXJfZmxvYXQnKSkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgfSBlbHNlIHtcbiAgICBpZiAoaGFzRXh0ZW5zaW9uKGdsLCAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCcpKSB7XG4gICAgICByZXR1cm4gY3JlYXRlRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoZ2wpO1xuICAgIH1cblxuICAgIGNvbnN0IENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUID0gJ0VYVF9jb2xvcl9idWZmZXJfaGFsZl9mbG9hdCc7XG4gICAgaWYgKGhhc0V4dGVuc2lvbihnbCwgQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpKSB7XG4gICAgICBjb25zdCB0ZXh0dXJlSGFsZkZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICBnbC5nZXRFeHRlbnNpb24oQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpO1xuICAgICAgcmV0dXJuIGNyZWF0ZUhhbGZGbG9hdFRleHR1cmVBbmRCaW5kVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgICBnbCwgdGV4dHVyZUhhbGZGbG9hdEV4dGVuc2lvbik7XG4gICAgfVxuXG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgY29uc3QgaXNGcmFtZUJ1ZmZlckNvbXBsZXRlID0gY3JlYXRlRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoZ2wpO1xuICByZXR1cm4gaXNGcmFtZUJ1ZmZlckNvbXBsZXRlO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVGbG9hdFRleHR1cmVBbmRCaW5kVG9GcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTpcbiAgICBib29sZWFuIHtcbiAgY29uc3QgdGV4Q29uZmlnID0gZ2V0VGV4dHVyZUNvbmZpZyhnbCk7XG5cbiAgY29uc3QgdGV4dHVyZSA9IGdsLmNyZWF0ZVRleHR1cmUoKTtcbiAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSk7XG5cbiAgY29uc3Qgd2lkdGggPSAxO1xuICBjb25zdCBoZWlnaHQgPSAxO1xuICBnbC50ZXhJbWFnZTJEKFxuICAgICAgZ2wuVEVYVFVSRV8yRCwgMCwgdGV4Q29uZmlnLmludGVybmFsRm9ybWF0RmxvYXQsIHdpZHRoLCBoZWlnaHQsIDAsXG4gICAgICB0ZXhDb25maWcudGV4dHVyZUZvcm1hdEZsb2F0LCB0ZXhDb25maWcudGV4dHVyZVR5cGVGbG9hdCwgbnVsbCk7XG5cbiAgY29uc3QgZnJhbWVCdWZmZXIgPSBnbC5jcmVhdGVGcmFtZWJ1ZmZlcigpO1xuICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lQnVmZmVyKTtcbiAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRleHR1cmUsIDApO1xuXG4gIGNvbnN0IGlzRnJhbWVCdWZmZXJDb21wbGV0ZSA9XG4gICAgICBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKSA9PT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEU7XG5cbiAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCk7XG4gIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCk7XG4gIGdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSk7XG4gIGdsLmRlbGV0ZUZyYW1lYnVmZmVyKGZyYW1lQnVmZmVyKTtcblxuICByZXR1cm4gaXNGcmFtZUJ1ZmZlckNvbXBsZXRlO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVIYWxmRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoXG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb246IGFueSk6IGJvb2xlYW4ge1xuICBjb25zdCB0ZXhDb25maWcgPSBnZXRUZXh0dXJlQ29uZmlnKGdsLCB0ZXh0dXJlSGFsZkZsb2F0RXh0ZW5zaW9uKTtcbiAgY29uc3QgdGV4dHVyZSA9IGdsLmNyZWF0ZVRleHR1cmUoKTtcbiAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSk7XG5cbiAgY29uc3Qgd2lkdGggPSAxO1xuICBjb25zdCBoZWlnaHQgPSAxO1xuICBnbC50ZXhJbWFnZTJEKFxuICAgICAgZ2wuVEVYVFVSRV8yRCwgMCwgdGV4Q29uZmlnLmludGVybmFsRm9ybWF0SGFsZkZsb2F0LCB3aWR0aCwgaGVpZ2h0LCAwLFxuICAgICAgdGV4Q29uZmlnLnRleHR1cmVGb3JtYXRGbG9hdCwgdGV4Q29uZmlnLnRleHR1cmVUeXBlSGFsZkZsb2F0LCBudWxsKTtcblxuICBjb25zdCBmcmFtZUJ1ZmZlciA9IGdsLmNyZWF0ZUZyYW1lYnVmZmVyKCk7XG4gIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVCdWZmZXIpO1xuICBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChcbiAgICAgIGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSwgMCk7XG5cbiAgY29uc3QgaXNGcmFtZUJ1ZmZlckNvbXBsZXRlID1cbiAgICAgIGdsLmNoZWNrRnJhbWVidWZmZXJTdGF0dXMoZ2wuRlJBTUVCVUZGRVIpID09PSBnbC5GUkFNRUJVRkZFUl9DT01QTEVURTtcblxuICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKTtcbiAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKTtcbiAgZ2wuZGVsZXRlVGV4dHVyZSh0ZXh0dXJlKTtcbiAgZ2wuZGVsZXRlRnJhbWVidWZmZXIoZnJhbWVCdWZmZXIpO1xuXG4gIHJldHVybiBpc0ZyYW1lQnVmZmVyQ29tcGxldGU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1dlYkdMRmVuY2VFbmFibGVkKHdlYkdMVmVyc2lvbjogbnVtYmVyKSB7XG4gIGlmICh3ZWJHTFZlcnNpb24gIT09IDIpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGNvbnN0IGlzRW5hYmxlZCA9IChnbCBhcyBhbnkpLmZlbmNlU3luYyAhPSBudWxsO1xuICByZXR1cm4gaXNFbmFibGVkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0Tm90Q29tcGxleChcbiAgICB0ZW5zb3I6IFRlbnNvckluZm98VGVuc29ySW5mb1tdLCBvcE5hbWU6IHN0cmluZyk6IHZvaWQge1xuICBpZiAoIUFycmF5LmlzQXJyYXkodGVuc29yKSkge1xuICAgIHRlbnNvciA9IFt0ZW5zb3JdO1xuICB9XG4gIHRlbnNvci5mb3JFYWNoKHQgPT4ge1xuICAgIGlmICh0ICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIHQuZHR5cGUgIT09ICdjb21wbGV4NjQnLFxuICAgICAgICAgICgpID0+IGAke29wTmFtZX0gZG9lcyBub3Qgc3VwcG9ydCBjb21wbGV4NjQgdGVuc29ycyBgICtcbiAgICAgICAgICAgICAgJ2luIHRoZSBXZWJHTCBiYWNrZW5kLicpO1xuICAgIH1cbiAgfSk7XG59XG4iXX0=