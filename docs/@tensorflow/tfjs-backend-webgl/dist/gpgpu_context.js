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
import { getWebGLContext, setWebGLContext } from './canvas_util';
import * as gpgpu_util from './gpgpu_util';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';
export class GPGPUContext {
    constructor(gl) {
        this.outputTexture = null;
        this.program = null;
        this.disposed = false;
        this.itemsToPoll = [];
        const glVersion = env().getNumber('WEBGL_VERSION');
        if (gl != null) {
            this.gl = gl;
            setWebGLContext(glVersion, gl);
        }
        else {
            this.gl = getWebGLContext(glVersion);
        }
        gl = this.gl;
        if (env().getNumber('WEBGL_VERSION') === 2) {
            const gl2 = gl;
            this.createVertexArray = () => {
                return webgl_util.callAndCheck(gl2, () => gl2.createVertexArray());
            };
            this.bindVertexArray = (vao) => {
                return webgl_util.callAndCheck(gl2, () => gl2.bindVertexArray(vao));
            };
            this.deleteVertexArray = (vao) => {
                return webgl_util.callAndCheck(gl2, () => gl2.deleteVertexArray(vao));
            };
            this.getVertexArray = () => {
                return webgl_util.callAndCheck(gl2, () => gl2.getParameter(gl2.VERTEX_ARRAY_BINDING));
            };
        }
        else if (gl != null) {
            const ext = gl.getExtension('OES_vertex_array_object');
            if (ext == null) {
                throw new Error('All WebGL1 implementations are expected to offer' +
                    ' OES_vertex_array_object.');
            }
            this.createVertexArray = () => {
                return webgl_util.callAndCheck(gl, () => ext.createVertexArrayOES());
            };
            this.bindVertexArray = (vao) => {
                return webgl_util.callAndCheck(gl, () => ext.bindVertexArrayOES(vao));
            };
            this.deleteVertexArray = (vao) => {
                return webgl_util.callAndCheck(gl, () => ext.deleteVertexArrayOES(vao));
            };
            this.getVertexArray = () => {
                return webgl_util.callAndCheck(gl, () => gl.getParameter(ext.VERTEX_ARRAY_BINDING_OES));
            };
        }
        // WebGL 2.0 enables texture floats without an extension.
        let COLOR_BUFFER_FLOAT = 'WEBGL_color_buffer_float';
        const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
        this.parallelCompilationExtension =
            this.gl.getExtension('KHR_parallel_shader_compile');
        if (env().getNumber('WEBGL_VERSION') === 1) {
            const TEXTURE_FLOAT = 'OES_texture_float';
            const TEXTURE_HALF_FLOAT = 'OES_texture_half_float';
            this.textureFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, TEXTURE_FLOAT);
            if (webgl_util.hasExtension(this.gl, TEXTURE_HALF_FLOAT)) {
                this.textureHalfFloatExtension =
                    webgl_util.getExtensionOrThrow(this.gl, TEXTURE_HALF_FLOAT);
            }
            else if (env().get('WEBGL_FORCE_F16_TEXTURES')) {
                throw new Error('GL context does not support half float textures, yet the ' +
                    'environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
            }
            this.colorBufferFloatExtension = this.gl.getExtension(COLOR_BUFFER_FLOAT);
            if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                this.colorBufferHalfFloatExtension =
                    webgl_util.getExtensionOrThrow(this.gl, COLOR_BUFFER_HALF_FLOAT);
            }
            else if (env().get('WEBGL_FORCE_F16_TEXTURES')) {
                throw new Error('GL context does not support color renderable half floats, yet ' +
                    'the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
            }
        }
        else {
            COLOR_BUFFER_FLOAT = 'EXT_color_buffer_float';
            if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_FLOAT)) {
                this.colorBufferFloatExtension =
                    this.gl.getExtension(COLOR_BUFFER_FLOAT);
            }
            else if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                this.colorBufferHalfFloatExtension =
                    this.gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
            }
            else {
                throw new Error('GL context does not support color renderable floats');
            }
        }
        this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
        this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
        this.framebuffer = webgl_util.createFramebuffer(this.gl);
        this.textureConfig =
            tex_util.getTextureConfig(this.gl, this.textureHalfFloatExtension);
    }
    get debug() {
        return env().getBool('DEBUG');
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        if (this.program != null) {
            console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                ' This is probably a resource leak, delete the program with ' +
                'GPGPUContext.deleteProgram before disposing.');
        }
        if (this.outputTexture != null) {
            console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                'texture.  This is probably a resource leak, delete the output ' +
                'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                'disposing.');
        }
        const gl = this.gl;
        webgl_util.callAndCheck(gl, () => gl.finish());
        webgl_util.callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.deleteFramebuffer(this.framebuffer));
        webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.deleteBuffer(this.indexBuffer));
        this.disposed = true;
    }
    createFloat32MatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat32MatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createFloat16MatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat16MatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createUnsignedBytesMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createUnsignedBytesMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    uploadPixelDataToTexture(texture, pixels) {
        this.throwIfDisposed();
        gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
    }
    uploadDenseMatrixToTexture(texture, width, height, data) {
        this.throwIfDisposed();
        gpgpu_util.uploadDenseMatrixToTexture(this.gl, texture, width, height, data, this.textureConfig);
    }
    createFloat16PackedMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat16PackedMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createPackedMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    deleteMatrixTexture(texture) {
        this.throwIfDisposed();
        if (this.outputTexture === texture) {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
            this.outputTexture = null;
        }
        webgl_util.callAndCheck(this.gl, () => this.gl.deleteTexture(texture));
    }
    downloadByteEncodedFloatMatrixFromOutputTexture(texture, rows, columns) {
        return this.downloadMatrixDriver(texture, () => gpgpu_util.downloadByteEncodedFloatMatrixFromOutputTexture(this.gl, rows, columns, this.textureConfig));
    }
    downloadPackedMatrixFromBuffer(buffer, batch, rows, columns, physicalRows, physicalCols) {
        return gpgpu_util.downloadPackedMatrixFromBuffer(this.gl, buffer, batch, rows, columns, physicalRows, physicalCols, this.textureConfig);
    }
    downloadFloat32MatrixFromBuffer(buffer, size) {
        return gpgpu_util.downloadFloat32MatrixFromBuffer(this.gl, buffer, size);
    }
    createBufferFromTexture(texture, rows, columns) {
        this.bindTextureToFrameBuffer(texture);
        const result = gpgpu_util.createBufferFromOutputTexture(this.gl, rows, columns, this.textureConfig);
        this.unbindTextureToFrameBuffer();
        return result;
    }
    createAndWaitForFence() {
        const fenceContext = this.createFence(this.gl);
        return this.pollFence(fenceContext);
    }
    createFence(gl) {
        let query;
        let isFencePassed;
        if (env().getBool('WEBGL_FENCE_API_ENABLED')) {
            const gl2 = gl;
            const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
            gl.flush();
            isFencePassed = () => {
                const status = gl2.clientWaitSync(sync, 0, 0);
                return status === gl2.ALREADY_SIGNALED ||
                    status === gl2.CONDITION_SATISFIED;
            };
            query = sync;
        }
        else if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
            query = this.beginQuery();
            this.endQuery();
            isFencePassed = () => this.isQueryAvailable(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
        }
        else {
            // If we have no way to fence, return true immediately. This will fire in
            // WebGL 1.0 when there is no disjoint query timer. In this case, because
            // the fence passes immediately, we'll immediately ask for a download of
            // the texture, which will cause the UI thread to hang.
            isFencePassed = () => true;
        }
        return { query, isFencePassed };
    }
    downloadMatrixFromPackedTexture(texture, physicalRows, physicalCols) {
        return this.downloadMatrixDriver(texture, () => gpgpu_util.downloadMatrixFromPackedOutputTexture(this.gl, physicalRows, physicalCols));
    }
    createProgram(fragmentShader) {
        this.throwIfDisposed();
        const gl = this.gl;
        if (this.vertexShader == null) {
            this.vertexShader = gpgpu_util.createVertexShader(gl);
        }
        const program = webgl_util.createProgram(gl);
        webgl_util.callAndCheck(gl, () => gl.attachShader(program, this.vertexShader));
        webgl_util.callAndCheck(gl, () => gl.attachShader(program, fragmentShader));
        webgl_util.linkProgram(gl, program);
        let program2;
        {
            program2 = Object.assign(program, {
                vao: this.createVertexArray(),
            });
            this.bindVertexArray(program2.vao);
            // Bind index buffer, and vertex buffers based on program attrib
            // locations.
            webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer));
            console.assert(gpgpu_util.bindVertexProgramAttributeStreams(gl, program2, this.vertexBuffer), 'gpgpu_util.bindVertexProgramAttributeStreams not fully successful.');
            if (this.debug) {
                webgl_util.validateProgram(gl, program2);
            }
        }
        this.setProgram(program2);
        return program2;
    }
    deleteProgram(program) {
        this.throwIfDisposed();
        if (program === this.program) {
            this.program = null;
        }
        if (program != null) {
            webgl_util.callAndCheck(this.gl, () => this.gl.deleteProgram(program));
            this.deleteVertexArray(program.vao);
        }
    }
    setProgram(program) {
        this.throwIfDisposed();
        this.program = program;
        if (this.program != null) {
            this.bindVertexArray(this.program.vao);
            if (this.debug) {
                webgl_util.validateProgram(this.gl, this.program);
            }
        }
        webgl_util.callAndCheck(this.gl, () => this.gl.useProgram(program));
    }
    getUniformLocation(program, uniformName, shouldThrow = true) {
        this.throwIfDisposed();
        if (shouldThrow) {
            return webgl_util.getProgramUniformLocationOrThrow(this.gl, program, uniformName);
        }
        else {
            return webgl_util.getProgramUniformLocation(this.gl, program, uniformName);
        }
    }
    getAttributeLocation(program, attribute) {
        this.throwIfDisposed();
        return webgl_util.callAndCheck(this.gl, () => this.gl.getAttribLocation(program, attribute));
    }
    getUniformLocationNoThrow(program, uniformName) {
        this.throwIfDisposed();
        return this.gl.getUniformLocation(program, uniformName);
    }
    setInputMatrixTexture(inputMatrixTexture, uniformLocation, textureUnit) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        webgl_util.bindTextureToProgramUniformSampler(this.gl, inputMatrixTexture, uniformLocation, textureUnit);
    }
    setOutputMatrixTexture(outputMatrixTexture, rows, columns) {
        this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
    }
    setOutputPackedMatrixTexture(outputPackedMatrixTexture, rows, columns) {
        this.throwIfDisposed();
        const [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
        this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
    }
    setOutputMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
        this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
    }
    setOutputPackedMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
        throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
    }
    debugValidate() {
        if (this.program != null) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.validateFramebuffer(this.gl);
    }
    executeProgram() {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        const gl = this.gl;
        if (this.debug) {
            const boundVao = this.getVertexArray();
            console.assert(boundVao === this.program.vao, 'VAO changed between setProgram and executeProgram!');
            this.debugValidate();
        }
        webgl_util.callAndCheck(gl, () => gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0));
    }
    blockUntilAllProgramsCompleted() {
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, () => this.gl.finish());
    }
    getQueryTimerExtension() {
        if (this.disjointQueryTimerExtension == null) {
            this.disjointQueryTimerExtension =
                webgl_util.getExtensionOrThrow(this.gl, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2 ?
                    'EXT_disjoint_timer_query_webgl2' :
                    'EXT_disjoint_timer_query');
        }
        return this.disjointQueryTimerExtension;
    }
    getQueryTimerExtensionWebGL2() {
        return this.getQueryTimerExtension();
    }
    getQueryTimerExtensionWebGL1() {
        return this.getQueryTimerExtension();
    }
    beginQuery() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            const query = gl2.createQuery();
            gl2.beginQuery(ext.TIME_ELAPSED_EXT, query);
            return query;
        }
        const ext = this.getQueryTimerExtensionWebGL1();
        const query = ext.createQueryEXT();
        ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
        return query;
    }
    endQuery() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            gl2.endQuery(ext.TIME_ELAPSED_EXT);
            return;
        }
        const ext = this.getQueryTimerExtensionWebGL1();
        ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
    }
    async waitForQueryAndGetTime(query) {
        await util.repeatedTry(() => this.disposed || // while testing contexts are created / disposed
            // in rapid succession, so without this check we
            // may poll for the query timer indefinitely
            this.isQueryAvailable(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')));
        return this.getQueryTime(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
    }
    getQueryTime(query, queryTimerVersion) {
        if (queryTimerVersion === 0) {
            return null;
        }
        if (queryTimerVersion === 2) {
            const gl2 = this.gl;
            const timeElapsedNanos = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
            // Return milliseconds.
            return timeElapsedNanos / 1000000;
        }
        else {
            const ext = this.getQueryTimerExtensionWebGL1();
            const timeElapsedNanos = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
            // Return milliseconds.
            return timeElapsedNanos / 1000000;
        }
    }
    isQueryAvailable(query, queryTimerVersion) {
        if (queryTimerVersion === 0) {
            return true;
        }
        if (queryTimerVersion === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            const available = gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE);
            if (this.disjoint == null) {
                this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
            }
            return available && !this.disjoint;
        }
        else {
            const ext = this.getQueryTimerExtensionWebGL1();
            const available = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);
            if (this.disjoint == null) {
                this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
            }
            return available && !this.disjoint;
        }
    }
    pollFence(fenceContext) {
        return new Promise(resolve => {
            this.addItemToPoll(() => fenceContext.isFencePassed(), () => resolve());
        });
    }
    pollItems() {
        // Find the last query that has finished.
        const index = linearSearchLastTrue(this.itemsToPoll.map(x => x.isDoneFn));
        for (let i = 0; i <= index; ++i) {
            const { resolveFn } = this.itemsToPoll[i];
            resolveFn();
        }
        this.itemsToPoll = this.itemsToPoll.slice(index + 1);
    }
    addItemToPoll(isDoneFn, resolveFn) {
        this.itemsToPoll.push({ isDoneFn, resolveFn });
        if (this.itemsToPoll.length > 1) {
            // We already have a running loop that polls.
            return;
        }
        // Start a new loop that polls.
        let scheduleFn = undefined;
        if ('setTimeoutCustom' in env().platform) {
            scheduleFn = env().platform.setTimeoutCustom.bind(env().platform);
        }
        util.repeatedTry(() => {
            this.pollItems();
            // End the loop if no more items to poll.
            return this.itemsToPoll.length === 0;
        }, () => 0, null, scheduleFn);
    }
    bindTextureToFrameBuffer(texture) {
        this.throwIfDisposed();
        webgl_util.bindColorTextureToFramebuffer(this.gl, texture, this.framebuffer);
        if (this.debug) {
            webgl_util.validateFramebuffer(this.gl);
        }
    }
    unbindTextureToFrameBuffer() {
        if (this.outputTexture != null) {
            webgl_util.bindColorTextureToFramebuffer(this.gl, this.outputTexture, this.framebuffer);
            if (this.debug) {
                webgl_util.validateFramebuffer(this.gl);
            }
        }
        else {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
        }
    }
    downloadMatrixDriver(texture, downloadAndDecode) {
        this.bindTextureToFrameBuffer(texture);
        const result = downloadAndDecode();
        this.unbindTextureToFrameBuffer();
        return result;
    }
    setOutputMatrixTextureDriver(outputMatrixTextureMaybePacked, width, height) {
        this.throwIfDisposed();
        const gl = this.gl;
        webgl_util.bindColorTextureToFramebuffer(gl, outputMatrixTextureMaybePacked, this.framebuffer);
        if (this.debug) {
            webgl_util.validateFramebuffer(gl);
        }
        this.outputTexture = outputMatrixTextureMaybePacked;
        webgl_util.callAndCheck(gl, () => gl.viewport(0, 0, width, height));
        webgl_util.callAndCheck(gl, () => gl.scissor(0, 0, width, height));
    }
    setOutputMatrixWriteRegionDriver(x, y, width, height) {
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, () => this.gl.scissor(x, y, width, height));
    }
    throwIfDisposed() {
        if (this.disposed) {
            throw new Error('Attempted to use disposed GPGPUContext.');
        }
    }
    throwIfNoProgram() {
        if (this.program == null) {
            throw new Error('No GPU program is currently set.');
        }
    }
}
/**
 * Finds the index of the last true element using linear search.
 * Note: We can't do binary search because Chrome expects us to explicitly
 * test all fences before download:
 * https://github.com/tensorflow/tfjs/issues/1145
 */
export function linearSearchLastTrue(arr) {
    let i = 0;
    for (; i < arr.length; ++i) {
        const isDone = arr[i]();
        if (!isDone) {
            break;
        }
    }
    return i - 1;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3BncHVfY29udGV4dC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvZ3BncHVfY29udGV4dC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUF5QixJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV2RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGVBQWUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUMvRCxPQUFPLEtBQUssVUFBVSxNQUFNLGNBQWMsQ0FBQztBQUMzQyxPQUFPLEtBQUssUUFBUSxNQUFNLFlBQVksQ0FBQztBQUd2QyxPQUFPLEtBQUssVUFBVSxNQUFNLGNBQWMsQ0FBQztBQWEzQyxNQUFNLE9BQU8sWUFBWTtJQXdCdkIsWUFBWSxFQUEwQjtRQVp0QyxrQkFBYSxHQUFzQixJQUFJLENBQUM7UUFDeEMsWUFBTyxHQUE2QixJQUFJLENBQUM7UUFDakMsYUFBUSxHQUFHLEtBQUssQ0FBQztRQXFpQmpCLGdCQUFXLEdBQWUsRUFBRSxDQUFDO1FBMWhCbkMsTUFBTSxTQUFTLEdBQUcsR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLGVBQWUsQ0FBQyxDQUFDO1FBQ25ELElBQUksRUFBRSxJQUFJLElBQUksRUFBRTtZQUNkLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDO1lBQ2IsZUFBZSxDQUFDLFNBQVMsRUFBRSxFQUFFLENBQUMsQ0FBQztTQUNoQzthQUFNO1lBQ0wsSUFBSSxDQUFDLEVBQUUsR0FBRyxlQUFlLENBQUMsU0FBUyxDQUFDLENBQUM7U0FDdEM7UUFDRCxFQUFFLEdBQUcsSUFBSSxDQUFDLEVBQUUsQ0FBQztRQUViLElBQUksR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLGVBQWUsQ0FBQyxLQUFLLENBQUMsRUFBRTtZQUMxQyxNQUFNLEdBQUcsR0FBRyxFQUE0QixDQUFDO1lBQ3pD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QztRQUN4QixJQUFJLENBQUMsTUFBTSxFQUFFO1lBQ1gsTUFBTTtTQUNQO0tBQ0Y7SUFDRCxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUM7QUFDZixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2VudiwgUGl4ZWxEYXRhLCBUeXBlZEFycmF5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge2dldFdlYkdMQ29udGV4dCwgc2V0V2ViR0xDb250ZXh0fSBmcm9tICcuL2NhbnZhc191dGlsJztcbmltcG9ydCAqIGFzIGdwZ3B1X3V0aWwgZnJvbSAnLi9ncGdwdV91dGlsJztcbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0IHtUZXh0dXJlLCBUZXh0dXJlQ29uZmlnfSBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7V2ViR0wxRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uLCBXZWJHTDJEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb24sIFdlYkdMUGFyYWxsZWxDb21waWxhdGlvbkV4dGVuc2lvbn0gZnJvbSAnLi93ZWJnbF90eXBlcyc7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgRmVuY2VDb250ZXh0IHtcbiAgcXVlcnk6IFdlYkdMUXVlcnl8V2ViR0xTeW5jO1xuICBpc0ZlbmNlUGFzc2VkKCk6IGJvb2xlYW47XG59XG5cbnR5cGUgV2ViR0xWYW8gPSBXZWJHTFZlcnRleEFycmF5T2JqZWN0IHwgV2ViR0xWZXJ0ZXhBcnJheU9iamVjdE9FUztcblxuZXhwb3J0IGludGVyZmFjZSBHUEdQVUNvbnRleHRQcm9ncmFtIGV4dGVuZHMgV2ViR0xQcm9ncmFtIHtcbiAgdmFvOiBXZWJHTFZhbztcbn1cblxuZXhwb3J0IGNsYXNzIEdQR1BVQ29udGV4dCB7XG4gIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIHRleHR1cmVGbG9hdEV4dGVuc2lvbjoge307XG4gIHRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb246IHt9O1xuICBjb2xvckJ1ZmZlckZsb2F0RXh0ZW5zaW9uOiB7fTtcbiAgY29sb3JCdWZmZXJIYWxmRmxvYXRFeHRlbnNpb246IHt9O1xuICBkaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb246IFdlYkdMMkRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbnxcbiAgICAgIFdlYkdMMURpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbjtcbiAgcGFyYWxsZWxDb21waWxhdGlvbkV4dGVuc2lvbjogV2ViR0xQYXJhbGxlbENvbXBpbGF0aW9uRXh0ZW5zaW9uO1xuICB2ZXJ0ZXhCdWZmZXI6IFdlYkdMQnVmZmVyO1xuICBpbmRleEJ1ZmZlcjogV2ViR0xCdWZmZXI7XG4gIGZyYW1lYnVmZmVyOiBXZWJHTEZyYW1lYnVmZmVyO1xuICBvdXRwdXRUZXh0dXJlOiBXZWJHTFRleHR1cmV8bnVsbCA9IG51bGw7XG4gIHByb2dyYW06IEdQR1BVQ29udGV4dFByb2dyYW18bnVsbCA9IG51bGw7XG4gIHByaXZhdGUgZGlzcG9zZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSBkaXNqb2ludDogYm9vbGVhbjtcbiAgcHJpdmF0ZSB2ZXJ0ZXhTaGFkZXI6IFdlYkdMU2hhZGVyO1xuICB0ZXh0dXJlQ29uZmlnOiBUZXh0dXJlQ29uZmlnO1xuXG4gIGNyZWF0ZVZlcnRleEFycmF5OiAoKSA9PiBXZWJHTFZhbyB8IG51bGw7XG4gIGJpbmRWZXJ0ZXhBcnJheTogKHZhbzogV2ViR0xWYW8gfCBudWxsKSA9PiB2b2lkO1xuICBkZWxldGVWZXJ0ZXhBcnJheTogKHZhbzogV2ViR0xWYW8gfCBudWxsKSA9PiB2b2lkO1xuICBnZXRWZXJ0ZXhBcnJheTogKCkgPT4gV2ViR0xWYW8gfCBudWxsO1xuXG4gIGNvbnN0cnVjdG9yKGdsPzogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gICAgY29uc3QgZ2xWZXJzaW9uID0gZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9WRVJTSU9OJyk7XG4gICAgaWYgKGdsICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZ2wgPSBnbDtcbiAgICAgIHNldFdlYkdMQ29udGV4dChnbFZlcnNpb24sIGdsKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5nbCA9IGdldFdlYkdMQ29udGV4dChnbFZlcnNpb24pO1xuICAgIH1cbiAgICBnbCA9IHRoaXMuZ2w7XG5cbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9WRVJTSU9OJykgPT09IDIpIHtcbiAgICAgIGNvbnN0IGdsMiA9IGdsIGFzIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQ7XG4gICAgICB0aGlzLmNyZWF0ZVZlcnRleEFycmF5ID0gKCkgPT4ge1xuICAgICAgICByZXR1cm4gd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wyLFxuICAgICAgICAgICgpID0+IGdsMi5jcmVhdGVWZXJ0ZXhBcnJheSgpKTtcbiAgICAgIH07XG4gICAgICB0aGlzLmJpbmRWZXJ0ZXhBcnJheSA9ICh2YW86IFdlYkdMVmFvfG51bGwpID0+IHtcbiAgICAgICAgcmV0dXJuIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsMixcbiAgICAgICAgICAoKSA9PiBnbDIuYmluZFZlcnRleEFycmF5KHZhbyBhcyBXZWJHTFZlcnRleEFycmF5T2JqZWN0KSk7XG4gICAgICB9O1xuICAgICAgdGhpcy5kZWxldGVWZXJ0ZXhBcnJheSA9ICh2YW86IFdlYkdMVmFvfG51bGwpID0+IHtcbiAgICAgICAgcmV0dXJuIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsMixcbiAgICAgICAgICAoKSA9PiBnbDIuZGVsZXRlVmVydGV4QXJyYXkodmFvIGFzIFdlYkdMVmVydGV4QXJyYXlPYmplY3QpKTtcbiAgICAgIH07XG4gICAgICB0aGlzLmdldFZlcnRleEFycmF5ID0gKCkgPT4ge1xuICAgICAgICByZXR1cm4gd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wyLFxuICAgICAgICAgICgpID0+IGdsMi5nZXRQYXJhbWV0ZXIoZ2wyLlZFUlRFWF9BUlJBWV9CSU5ESU5HKSk7XG4gICAgICB9O1xuICAgIH0gZWxzZSBpZiAoZ2wgIT0gbnVsbCkge1xuICAgICAgY29uc3QgZXh0ID0gZ2wuZ2V0RXh0ZW5zaW9uKCdPRVNfdmVydGV4X2FycmF5X29iamVjdCcpO1xuICAgICAgaWYgKGV4dCA9PSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdBbGwgV2ViR0wxIGltcGxlbWVudGF0aW9ucyBhcmUgZXhwZWN0ZWQgdG8gb2ZmZXInICtcbiAgICAgICAgICAgICcgT0VTX3ZlcnRleF9hcnJheV9vYmplY3QuJyk7XG4gICAgICB9XG4gICAgICB0aGlzLmNyZWF0ZVZlcnRleEFycmF5ID0gKCkgPT4ge1xuICAgICAgICByZXR1cm4gd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsXG4gICAgICAgICAgKCkgPT4gZXh0LmNyZWF0ZVZlcnRleEFycmF5T0VTKCkpO1xuICAgICAgfTtcbiAgICAgIHRoaXMuYmluZFZlcnRleEFycmF5ID0gKHZhbzogV2ViR0xWYW98bnVsbCkgPT4ge1xuICAgICAgICByZXR1cm4gd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsXG4gICAgICAgICAgKCkgPT4gZXh0LmJpbmRWZXJ0ZXhBcnJheU9FUyh2YW8gYXMgV2ViR0xWZXJ0ZXhBcnJheU9iamVjdE9FUykpO1xuICAgICAgfTtcbiAgICAgIHRoaXMuZGVsZXRlVmVydGV4QXJyYXkgPSAodmFvOiBXZWJHTFZhb3xudWxsKSA9PiB7XG4gICAgICAgIHJldHVybiB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCxcbiAgICAgICAgICAoKSA9PiBleHQuZGVsZXRlVmVydGV4QXJyYXlPRVModmFvIGFzIFdlYkdMVmVydGV4QXJyYXlPYmplY3RPRVMpKTtcbiAgICAgIH07XG4gICAgICB0aGlzLmdldFZlcnRleEFycmF5ID0gKCkgPT4ge1xuICAgICAgICByZXR1cm4gd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsXG4gICAgICAgICAgKCkgPT4gZ2wuZ2V0UGFyYW1ldGVyKGV4dC5WRVJURVhfQVJSQVlfQklORElOR19PRVMpKTtcbiAgICAgIH07XG4gICAgfVxuXG4gICAgLy8gV2ViR0wgMi4wIGVuYWJsZXMgdGV4dHVyZSBmbG9hdHMgd2l0aG91dCBhbiBleHRlbnNpb24uXG4gICAgbGV0IENPTE9SX0JVRkZFUl9GTE9BVCA9ICdXRUJHTF9jb2xvcl9idWZmZXJfZmxvYXQnO1xuICAgIGNvbnN0IENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUID0gJ0VYVF9jb2xvcl9idWZmZXJfaGFsZl9mbG9hdCc7XG4gICAgdGhpcy5wYXJhbGxlbENvbXBpbGF0aW9uRXh0ZW5zaW9uID1cbiAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oJ0tIUl9wYXJhbGxlbF9zaGFkZXJfY29tcGlsZScpO1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSA9PT0gMSkge1xuICAgICAgY29uc3QgVEVYVFVSRV9GTE9BVCA9ICdPRVNfdGV4dHVyZV9mbG9hdCc7XG4gICAgICBjb25zdCBURVhUVVJFX0hBTEZfRkxPQVQgPSAnT0VTX3RleHR1cmVfaGFsZl9mbG9hdCc7XG5cbiAgICAgIHRoaXMudGV4dHVyZUZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgVEVYVFVSRV9GTE9BVCk7XG4gICAgICBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgVEVYVFVSRV9IQUxGX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLnRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsIFRFWFRVUkVfSEFMRl9GTE9BVCk7XG4gICAgICB9IGVsc2UgaWYgKGVudigpLmdldCgnV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTJykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0dMIGNvbnRleHQgZG9lcyBub3Qgc3VwcG9ydCBoYWxmIGZsb2F0IHRleHR1cmVzLCB5ZXQgdGhlICcgK1xuICAgICAgICAgICAgJ2Vudmlyb25tZW50IGZsYWcgV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTIGlzIHNldCB0byB0cnVlLicpO1xuICAgICAgfVxuXG4gICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPSB0aGlzLmdsLmdldEV4dGVuc2lvbihDT0xPUl9CVUZGRVJfRkxPQVQpO1xuICAgICAgaWYgKHdlYmdsX3V0aWwuaGFzRXh0ZW5zaW9uKHRoaXMuZ2wsIENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLmNvbG9yQnVmZmVySGFsZkZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCBDT0xPUl9CVUZGRVJfSEFMRl9GTE9BVCk7XG4gICAgICB9IGVsc2UgaWYgKGVudigpLmdldCgnV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTJykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0dMIGNvbnRleHQgZG9lcyBub3Qgc3VwcG9ydCBjb2xvciByZW5kZXJhYmxlIGhhbGYgZmxvYXRzLCB5ZXQgJyArXG4gICAgICAgICAgICAndGhlIGVudmlyb25tZW50IGZsYWcgV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTIGlzIHNldCB0byB0cnVlLicpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBDT0xPUl9CVUZGRVJfRkxPQVQgPSAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCc7XG4gICAgICBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgQ09MT1JfQlVGRkVSX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oQ09MT1JfQlVGRkVSX0ZMT0FUKTtcbiAgICAgIH0gZWxzZSBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpKSB7XG4gICAgICAgIHRoaXMuY29sb3JCdWZmZXJIYWxmRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdHTCBjb250ZXh0IGRvZXMgbm90IHN1cHBvcnQgY29sb3IgcmVuZGVyYWJsZSBmbG9hdHMnKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLnZlcnRleEJ1ZmZlciA9IGdwZ3B1X3V0aWwuY3JlYXRlVmVydGV4QnVmZmVyKHRoaXMuZ2wpO1xuICAgIHRoaXMuaW5kZXhCdWZmZXIgPSBncGdwdV91dGlsLmNyZWF0ZUluZGV4QnVmZmVyKHRoaXMuZ2wpO1xuICAgIHRoaXMuZnJhbWVidWZmZXIgPSB3ZWJnbF91dGlsLmNyZWF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuXG4gICAgdGhpcy50ZXh0dXJlQ29uZmlnID1cbiAgICAgICAgdGV4X3V0aWwuZ2V0VGV4dHVyZUNvbmZpZyh0aGlzLmdsLCB0aGlzLnRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb24pO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXQgZGVidWcoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGVudigpLmdldEJvb2woJ0RFQlVHJyk7XG4gIH1cblxuICBwdWJsaWMgZGlzcG9zZSgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAodGhpcy5wcm9ncmFtICE9IG51bGwpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnRGlzcG9zaW5nIGEgR1BHUFVDb250ZXh0IHRoYXQgc3RpbGwgaGFzIGEgYm91bmQgV2ViR0xQcm9ncmFtLicgK1xuICAgICAgICAgICcgVGhpcyBpcyBwcm9iYWJseSBhIHJlc291cmNlIGxlYWssIGRlbGV0ZSB0aGUgcHJvZ3JhbSB3aXRoICcgK1xuICAgICAgICAgICdHUEdQVUNvbnRleHQuZGVsZXRlUHJvZ3JhbSBiZWZvcmUgZGlzcG9zaW5nLicpO1xuICAgIH1cbiAgICBpZiAodGhpcy5vdXRwdXRUZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnRGlzcG9zaW5nIGEgR1BHUFVDb250ZXh0IHRoYXQgc3RpbGwgaGFzIGEgYm91bmQgb3V0cHV0IG1hdHJpeCAnICtcbiAgICAgICAgICAndGV4dHVyZS4gIFRoaXMgaXMgcHJvYmFibHkgYSByZXNvdXJjZSBsZWFrLCBkZWxldGUgdGhlIG91dHB1dCAnICtcbiAgICAgICAgICAnbWF0cml4IHRleHR1cmUgd2l0aCBHUEdQVUNvbnRleHQuZGVsZXRlTWF0cml4VGV4dHVyZSBiZWZvcmUgJyArXG4gICAgICAgICAgJ2Rpc3Bvc2luZy4nKTtcbiAgICB9XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5maW5pc2goKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVGcmFtZWJ1ZmZlcih0aGlzLmZyYW1lYnVmZmVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUJ1ZmZlcih0aGlzLmluZGV4QnVmZmVyKSk7XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlRmxvYXQzMk1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZUZsb2F0MzJNYXRyaXhUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUZsb2F0MTZNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogVGV4dHVyZSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC5jcmVhdGVGbG9hdDE2TWF0cml4VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgcm93cywgY29sdW1ucywgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVVbnNpZ25lZEJ5dGVzTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6XG4gICAgICBUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZVVuc2lnbmVkQnl0ZXNNYXRyaXhUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZFBpeGVsRGF0YVRvVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICAgIHBpeGVsczogUGl4ZWxEYXRhfEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fFxuICAgICAgSW1hZ2VCaXRtYXApIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGdwZ3B1X3V0aWwudXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKHRoaXMuZ2wsIHRleHR1cmUsIHBpeGVscyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkRGVuc2VNYXRyaXhUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkYXRhOiBUeXBlZEFycmF5KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBncGdwdV91dGlsLnVwbG9hZERlbnNlTWF0cml4VG9UZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCB0ZXh0dXJlLCB3aWR0aCwgaGVpZ2h0LCBkYXRhLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUZsb2F0MTZQYWNrZWRNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTpcbiAgICAgIFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlRmxvYXQxNlBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMsIHRoaXMudGV4dHVyZUNvbmZpZyk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgcm93cywgY29sdW1ucywgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVNYXRyaXhUZXh0dXJlKHRleHR1cmU6IFdlYkdMVGV4dHVyZSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSA9PT0gdGV4dHVyZSkge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBudWxsO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSkpO1xuICB9XG5cbiAgcHVibGljIGRvd25sb2FkQnl0ZUVuY29kZWRGbG9hdE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRCeXRlRW5jb2RlZEZsb2F0TWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZFBhY2tlZE1hdHJpeEZyb21CdWZmZXIoXG4gICAgICBidWZmZXI6IFdlYkdMQnVmZmVyLCBiYXRjaDogbnVtYmVyLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICAgIHBoeXNpY2FsUm93czogbnVtYmVyLCBwaHlzaWNhbENvbHM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuZG93bmxvYWRQYWNrZWRNYXRyaXhGcm9tQnVmZmVyKFxuICAgICAgICB0aGlzLmdsLCBidWZmZXIsIGJhdGNoLCByb3dzLCBjb2x1bW5zLCBwaHlzaWNhbFJvd3MsIHBoeXNpY2FsQ29scyxcbiAgICAgICAgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZEZsb2F0MzJNYXRyaXhGcm9tQnVmZmVyKGJ1ZmZlcjogV2ViR0xCdWZmZXIsIHNpemU6IG51bWJlcik6XG4gICAgICBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiBncGdwdV91dGlsLmRvd25sb2FkRmxvYXQzMk1hdHJpeEZyb21CdWZmZXIodGhpcy5nbCwgYnVmZmVyLCBzaXplKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVCdWZmZXJGcm9tVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTEJ1ZmZlciB7XG4gICAgdGhpcy5iaW5kVGV4dHVyZVRvRnJhbWVCdWZmZXIodGV4dHVyZSk7XG4gICAgY29uc3QgcmVzdWx0ID0gZ3BncHVfdXRpbC5jcmVhdGVCdWZmZXJGcm9tT3V0cHV0VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICAgIHRoaXMudW5iaW5kVGV4dHVyZVRvRnJhbWVCdWZmZXIoKTtcbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUFuZFdhaXRGb3JGZW5jZSgpOiBQcm9taXNlPHZvaWQ+IHtcbiAgICBjb25zdCBmZW5jZUNvbnRleHQgPSB0aGlzLmNyZWF0ZUZlbmNlKHRoaXMuZ2wpO1xuICAgIHJldHVybiB0aGlzLnBvbGxGZW5jZShmZW5jZUNvbnRleHQpO1xuICB9XG5cbiAgcHJpdmF0ZSBjcmVhdGVGZW5jZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogRmVuY2VDb250ZXh0IHtcbiAgICBsZXQgcXVlcnk6IFdlYkdMUXVlcnl8V2ViR0xTeW5jO1xuICAgIGxldCBpc0ZlbmNlUGFzc2VkOiAoKSA9PiBib29sZWFuO1xuXG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX0ZFTkNFX0FQSV9FTkFCTEVEJykpIHtcbiAgICAgIGNvbnN0IGdsMiA9IGdsIGFzIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQ7XG5cbiAgICAgIGNvbnN0IHN5bmMgPSBnbDIuZmVuY2VTeW5jKGdsMi5TWU5DX0dQVV9DT01NQU5EU19DT01QTEVURSwgMCk7XG4gICAgICBnbC5mbHVzaCgpO1xuXG4gICAgICBpc0ZlbmNlUGFzc2VkID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBzdGF0dXMgPSBnbDIuY2xpZW50V2FpdFN5bmMoc3luYywgMCwgMCk7XG4gICAgICAgIHJldHVybiBzdGF0dXMgPT09IGdsMi5BTFJFQURZX1NJR05BTEVEIHx8XG4gICAgICAgICAgICBzdGF0dXMgPT09IGdsMi5DT05ESVRJT05fU0FUSVNGSUVEO1xuICAgICAgfTtcblxuICAgICAgcXVlcnkgPSBzeW5jO1xuICAgIH0gZWxzZSBpZiAoXG4gICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSA+IDApIHtcbiAgICAgIHF1ZXJ5ID0gdGhpcy5iZWdpblF1ZXJ5KCk7XG4gICAgICB0aGlzLmVuZFF1ZXJ5KCk7XG4gICAgICBpc0ZlbmNlUGFzc2VkID0gKCkgPT4gdGhpcy5pc1F1ZXJ5QXZhaWxhYmxlKFxuICAgICAgICAgIHF1ZXJ5LFxuICAgICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHdlIGhhdmUgbm8gd2F5IHRvIGZlbmNlLCByZXR1cm4gdHJ1ZSBpbW1lZGlhdGVseS4gVGhpcyB3aWxsIGZpcmUgaW5cbiAgICAgIC8vIFdlYkdMIDEuMCB3aGVuIHRoZXJlIGlzIG5vIGRpc2pvaW50IHF1ZXJ5IHRpbWVyLiBJbiB0aGlzIGNhc2UsIGJlY2F1c2VcbiAgICAgIC8vIHRoZSBmZW5jZSBwYXNzZXMgaW1tZWRpYXRlbHksIHdlJ2xsIGltbWVkaWF0ZWx5IGFzayBmb3IgYSBkb3dubG9hZCBvZlxuICAgICAgLy8gdGhlIHRleHR1cmUsIHdoaWNoIHdpbGwgY2F1c2UgdGhlIFVJIHRocmVhZCB0byBoYW5nLlxuICAgICAgaXNGZW5jZVBhc3NlZCA9ICgpID0+IHRydWU7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtxdWVyeSwgaXNGZW5jZVBhc3NlZH07XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcGh5c2ljYWxSb3dzOiBudW1iZXIsXG4gICAgICBwaHlzaWNhbENvbHM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkT3V0cHV0VGV4dHVyZShcbiAgICAgICAgICAgIHRoaXMuZ2wsIHBoeXNpY2FsUm93cywgcGh5c2ljYWxDb2xzKSk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUHJvZ3JhbShmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIpOiBHUEdQVUNvbnRleHRQcm9ncmFtIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICBpZiAodGhpcy52ZXJ0ZXhTaGFkZXIgPT0gbnVsbCkge1xuICAgICAgdGhpcy52ZXJ0ZXhTaGFkZXIgPSBncGdwdV91dGlsLmNyZWF0ZVZlcnRleFNoYWRlcihnbCk7XG4gICAgfVxuICAgIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9IHdlYmdsX3V0aWwuY3JlYXRlUHJvZ3JhbShnbCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgdGhpcy52ZXJ0ZXhTaGFkZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYXR0YWNoU2hhZGVyKHByb2dyYW0sIGZyYWdtZW50U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5saW5rUHJvZ3JhbShnbCwgcHJvZ3JhbSk7XG5cbiAgICBsZXQgcHJvZ3JhbTI6IEdQR1BVQ29udGV4dFByb2dyYW07XG4gICAge1xuICAgICAgcHJvZ3JhbTIgPSBPYmplY3QuYXNzaWduKHByb2dyYW0sIHtcbiAgICAgICAgdmFvOiB0aGlzLmNyZWF0ZVZlcnRleEFycmF5KCksXG4gICAgICB9KTtcbiAgICAgIHRoaXMuYmluZFZlcnRleEFycmF5KHByb2dyYW0yLnZhbyk7XG4gICAgICAvLyBCaW5kIGluZGV4IGJ1ZmZlciwgYW5kIHZlcnRleCBidWZmZXJzIGJhc2VkIG9uIHByb2dyYW0gYXR0cmliXG4gICAgICAvLyBsb2NhdGlvbnMuXG4gICAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgICBnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgdGhpcy5pbmRleEJ1ZmZlcikpO1xuICAgICAgY29uc29sZS5hc3NlcnQoXG4gICAgICAgIGdwZ3B1X3V0aWwuYmluZFZlcnRleFByb2dyYW1BdHRyaWJ1dGVTdHJlYW1zKGdsLCBwcm9ncmFtMixcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGhpcy52ZXJ0ZXhCdWZmZXIpLFxuICAgICAgICAnZ3BncHVfdXRpbC5iaW5kVmVydGV4UHJvZ3JhbUF0dHJpYnV0ZVN0cmVhbXMgbm90IGZ1bGx5IHN1Y2Nlc3NmdWwuJyk7XG5cbiAgICAgIGlmICh0aGlzLmRlYnVnKSB7XG4gICAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKGdsLCBwcm9ncmFtMik7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMuc2V0UHJvZ3JhbShwcm9ncmFtMik7XG5cbiAgICByZXR1cm4gcHJvZ3JhbTI7XG4gIH1cblxuICBwdWJsaWMgZGVsZXRlUHJvZ3JhbShwcm9ncmFtOiBHUEdQVUNvbnRleHRQcm9ncmFtKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBpZiAocHJvZ3JhbSA9PT0gdGhpcy5wcm9ncmFtKSB7XG4gICAgICB0aGlzLnByb2dyYW0gPSBudWxsO1xuICAgIH1cbiAgICBpZiAocHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSkpO1xuICAgICAgdGhpcy5kZWxldGVWZXJ0ZXhBcnJheShwcm9ncmFtLnZhbyk7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIHNldFByb2dyYW0ocHJvZ3JhbTogR1BHUFVDb250ZXh0UHJvZ3JhbXxudWxsKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnByb2dyYW0gPSBwcm9ncmFtO1xuXG4gICAgaWYgKHRoaXMucHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICB0aGlzLmJpbmRWZXJ0ZXhBcnJheSh0aGlzLnByb2dyYW0udmFvKTtcblxuICAgICAgaWYgKHRoaXMuZGVidWcpIHtcbiAgICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZVByb2dyYW0odGhpcy5nbCwgdGhpcy5wcm9ncmFtKTtcbiAgICAgIH1cbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC51c2VQcm9ncmFtKHByb2dyYW0pKTtcbiAgfVxuXG4gIHB1YmxpYyBnZXRVbmlmb3JtTG9jYXRpb24oXG4gICAgICBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIHVuaWZvcm1OYW1lOiBzdHJpbmcsXG4gICAgICBzaG91bGRUaHJvdyA9IHRydWUpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBpZiAoc2hvdWxkVGhyb3cpIHtcbiAgICAgIHJldHVybiB3ZWJnbF91dGlsLmdldFByb2dyYW1Vbmlmb3JtTG9jYXRpb25PclRocm93KFxuICAgICAgICAgIHRoaXMuZ2wsIHByb2dyYW0sIHVuaWZvcm1OYW1lKTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHdlYmdsX3V0aWwuZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbihcbiAgICAgICAgICB0aGlzLmdsLCBwcm9ncmFtLCB1bmlmb3JtTmFtZSk7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIGdldEF0dHJpYnV0ZUxvY2F0aW9uKHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYXR0cmlidXRlOiBzdHJpbmcpOlxuICAgICAgbnVtYmVyIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgdGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5nZXRBdHRyaWJMb2NhdGlvbihwcm9ncmFtLCBhdHRyaWJ1dGUpKTtcbiAgfVxuXG4gIHB1YmxpYyBnZXRVbmlmb3JtTG9jYXRpb25Ob1Rocm93KHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgdW5pZm9ybU5hbWU6IHN0cmluZyk6XG4gICAgICBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gdGhpcy5nbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgdW5pZm9ybU5hbWUpO1xuICB9XG5cbiAgcHVibGljIHNldElucHV0TWF0cml4VGV4dHVyZShcbiAgICAgIGlucHV0TWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCB1bmlmb3JtTG9jYXRpb246IFdlYkdMVW5pZm9ybUxvY2F0aW9uLFxuICAgICAgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVGV4dHVyZVRvUHJvZ3JhbVVuaWZvcm1TYW1wbGVyKFxuICAgICAgICB0aGlzLmdsLCBpbnB1dE1hdHJpeFRleHR1cmUsIHVuaWZvcm1Mb2NhdGlvbiwgdGV4dHVyZVVuaXQpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dE1hdHJpeFRleHR1cmUoXG4gICAgICBvdXRwdXRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKG91dHB1dE1hdHJpeFRleHR1cmUsIGNvbHVtbnMsIHJvd3MpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICBvdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFRleHR1cmVEcml2ZXIob3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZSwgd2lkdGgsIGhlaWdodCk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0TWF0cml4V3JpdGVSZWdpb24oXG4gICAgICBzdGFydFJvdzogbnVtYmVyLCBudW1Sb3dzOiBudW1iZXIsIHN0YXJ0Q29sdW1uOiBudW1iZXIsXG4gICAgICBudW1Db2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uRHJpdmVyKFxuICAgICAgICBzdGFydENvbHVtbiwgc3RhcnRSb3csIG51bUNvbHVtbnMsIG51bVJvd3MpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dFBhY2tlZE1hdHJpeFdyaXRlUmVnaW9uKFxuICAgICAgc3RhcnRSb3c6IG51bWJlciwgbnVtUm93czogbnVtYmVyLCBzdGFydENvbHVtbjogbnVtYmVyLFxuICAgICAgbnVtQ29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdzZXRPdXRwdXRQYWNrZWRNYXRyaXhXcml0ZVJlZ2lvbiBub3QgaW1wbGVtZW50ZWQuJyk7XG4gIH1cblxuICBwdWJsaWMgZGVidWdWYWxpZGF0ZSgpIHtcbiAgICBpZiAodGhpcy5wcm9ncmFtICE9IG51bGwpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgfVxuXG4gIHB1YmxpYyBleGVjdXRlUHJvZ3JhbSgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHRoaXMudGhyb3dJZk5vUHJvZ3JhbSgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICBpZiAodGhpcy5kZWJ1Zykge1xuICAgICAgY29uc3QgYm91bmRWYW8gPSB0aGlzLmdldFZlcnRleEFycmF5KCk7XG4gICAgICBjb25zb2xlLmFzc2VydChib3VuZFZhbyA9PT0gdGhpcy5wcm9ncmFtLnZhbyxcbiAgICAgICAgICAgICAgICAgICAgICdWQU8gY2hhbmdlZCBiZXR3ZWVuIHNldFByb2dyYW0gYW5kIGV4ZWN1dGVQcm9ncmFtIScpO1xuXG4gICAgICB0aGlzLmRlYnVnVmFsaWRhdGUoKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5kcmF3RWxlbWVudHMoZ2wuVFJJQU5HTEVTLCA2LCBnbC5VTlNJR05FRF9TSE9SVCwgMCkpO1xuICB9XG5cbiAgcHVibGljIGJsb2NrVW50aWxBbGxQcm9ncmFtc0NvbXBsZXRlZCgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuZmluaXNoKCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRRdWVyeVRpbWVyRXh0ZW5zaW9uKCk6IFdlYkdMMURpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvblxuICAgICAgfFdlYkdMMkRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbiB7XG4gICAgaWYgKHRoaXMuZGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uID09IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3coXG4gICAgICAgICAgICAgIHRoaXMuZ2wsXG4gICAgICAgICAgICAgIGVudigpLmdldE51bWJlcihcbiAgICAgICAgICAgICAgICAgICdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpID09PSAyID9cbiAgICAgICAgICAgICAgICAgICdFWFRfZGlzam9pbnRfdGltZXJfcXVlcnlfd2ViZ2wyJyA6XG4gICAgICAgICAgICAgICAgICAnRVhUX2Rpc2pvaW50X3RpbWVyX3F1ZXJ5JykgYXNcbiAgICAgICAgICAgICAgV2ViR0wxRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uIHxcbiAgICAgICAgICBXZWJHTDJEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb247XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbjtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMigpOiBXZWJHTDJEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb24ge1xuICAgIHJldHVybiB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb24oKTtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMSgpOiBXZWJHTDFEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb24ge1xuICAgIHJldHVybiB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb24oKSBhcyBXZWJHTDFEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb247XG4gIH1cblxuICBiZWdpblF1ZXJ5KCk6IFdlYkdMUXVlcnkge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9WRVJTSU9OJykgPT09IDIpIHtcbiAgICAgIGNvbnN0IGdsMiA9IHRoaXMuZ2wgYXMgV2ViR0wyUmVuZGVyaW5nQ29udGV4dDtcbiAgICAgIGNvbnN0IGV4dCA9IHRoaXMuZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMigpO1xuXG4gICAgICBjb25zdCBxdWVyeSA9IGdsMi5jcmVhdGVRdWVyeSgpO1xuICAgICAgZ2wyLmJlZ2luUXVlcnkoZXh0LlRJTUVfRUxBUFNFRF9FWFQsIHF1ZXJ5KTtcbiAgICAgIHJldHVybiBxdWVyeTtcbiAgICB9XG4gICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wxKCk7XG4gICAgY29uc3QgcXVlcnkgPSBleHQuY3JlYXRlUXVlcnlFWFQoKSBhcyBXZWJHTFF1ZXJ5O1xuICAgIGV4dC5iZWdpblF1ZXJ5RVhUKGV4dC5USU1FX0VMQVBTRURfRVhULCBxdWVyeSk7XG4gICAgcmV0dXJuIHF1ZXJ5O1xuICB9XG5cbiAgZW5kUXVlcnkoKSB7XG4gICAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSA9PT0gMikge1xuICAgICAgY29uc3QgZ2wyID0gdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0O1xuICAgICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wyKCk7XG4gICAgICBnbDIuZW5kUXVlcnkoZXh0LlRJTUVfRUxBUFNFRF9FWFQpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBleHQgPSB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb25XZWJHTDEoKTtcbiAgICBleHQuZW5kUXVlcnlFWFQoZXh0LlRJTUVfRUxBUFNFRF9FWFQpO1xuICB9XG5cbiAgcHVibGljIGFzeW5jIHdhaXRGb3JRdWVyeUFuZEdldFRpbWUocXVlcnk6IFdlYkdMUXVlcnkpOiBQcm9taXNlPG51bWJlcj4ge1xuICAgIGF3YWl0IHV0aWwucmVwZWF0ZWRUcnkoXG4gICAgICAgICgpID0+IHRoaXMuZGlzcG9zZWQgfHwgIC8vIHdoaWxlIHRlc3RpbmcgY29udGV4dHMgYXJlIGNyZWF0ZWQgLyBkaXNwb3NlZFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBpbiByYXBpZCBzdWNjZXNzaW9uLCBzbyB3aXRob3V0IHRoaXMgY2hlY2sgd2VcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gbWF5IHBvbGwgZm9yIHRoZSBxdWVyeSB0aW1lciBpbmRlZmluaXRlbHlcbiAgICAgICAgICAgIHRoaXMuaXNRdWVyeUF2YWlsYWJsZShcbiAgICAgICAgICAgICAgICBxdWVyeSxcbiAgICAgICAgICAgICAgICBlbnYoKS5nZXROdW1iZXIoXG4gICAgICAgICAgICAgICAgICAgICdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpKSk7XG4gICAgcmV0dXJuIHRoaXMuZ2V0UXVlcnlUaW1lKFxuICAgICAgICBxdWVyeSwgZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpKTtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lKHF1ZXJ5OiBXZWJHTFF1ZXJ5LCBxdWVyeVRpbWVyVmVyc2lvbjogbnVtYmVyKTogbnVtYmVyIHtcbiAgICBpZiAocXVlcnlUaW1lclZlcnNpb24gPT09IDApIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIGlmIChxdWVyeVRpbWVyVmVyc2lvbiA9PT0gMikge1xuICAgICAgY29uc3QgZ2wyID0gdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0O1xuXG4gICAgICBjb25zdCB0aW1lRWxhcHNlZE5hbm9zID0gZ2wyLmdldFF1ZXJ5UGFyYW1ldGVyKHF1ZXJ5LCBnbDIuUVVFUllfUkVTVUxUKTtcbiAgICAgIC8vIFJldHVybiBtaWxsaXNlY29uZHMuXG4gICAgICByZXR1cm4gdGltZUVsYXBzZWROYW5vcyAvIDEwMDAwMDA7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGV4dCA9IHRoaXMuZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMSgpO1xuXG4gICAgICBjb25zdCB0aW1lRWxhcHNlZE5hbm9zID1cbiAgICAgICAgICBleHQuZ2V0UXVlcnlPYmplY3RFWFQocXVlcnksIGV4dC5RVUVSWV9SRVNVTFRfRVhUKTtcbiAgICAgIC8vIFJldHVybiBtaWxsaXNlY29uZHMuXG4gICAgICByZXR1cm4gdGltZUVsYXBzZWROYW5vcyAvIDEwMDAwMDA7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBpc1F1ZXJ5QXZhaWxhYmxlKHF1ZXJ5OiBXZWJHTFF1ZXJ5LCBxdWVyeVRpbWVyVmVyc2lvbjogbnVtYmVyKTpcbiAgICAgIGJvb2xlYW4ge1xuICAgIGlmIChxdWVyeVRpbWVyVmVyc2lvbiA9PT0gMCkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgaWYgKHF1ZXJ5VGltZXJWZXJzaW9uID09PSAyKSB7XG4gICAgICBjb25zdCBnbDIgPSB0aGlzLmdsIGFzIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQ7XG4gICAgICBjb25zdCBleHQgPSB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb25XZWJHTDIoKTtcblxuICAgICAgY29uc3QgYXZhaWxhYmxlID1cbiAgICAgICAgICBnbDIuZ2V0UXVlcnlQYXJhbWV0ZXIocXVlcnksIGdsMi5RVUVSWV9SRVNVTFRfQVZBSUxBQkxFKTtcbiAgICAgIGlmICh0aGlzLmRpc2pvaW50ID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5kaXNqb2ludCA9IHRoaXMuZ2wuZ2V0UGFyYW1ldGVyKGV4dC5HUFVfRElTSk9JTlRfRVhUKTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGF2YWlsYWJsZSAmJiAhdGhpcy5kaXNqb2ludDtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wxKCk7XG5cbiAgICAgIGNvbnN0IGF2YWlsYWJsZSA9XG4gICAgICAgICAgZXh0LmdldFF1ZXJ5T2JqZWN0RVhUKHF1ZXJ5LCBleHQuUVVFUllfUkVTVUxUX0FWQUlMQUJMRV9FWFQpO1xuICAgICAgaWYgKHRoaXMuZGlzam9pbnQgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLmRpc2pvaW50ID0gdGhpcy5nbC5nZXRQYXJhbWV0ZXIoZXh0LkdQVV9ESVNKT0lOVF9FWFQpO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4gYXZhaWxhYmxlICYmICF0aGlzLmRpc2pvaW50O1xuICAgIH1cbiAgfVxuXG4gIHBvbGxGZW5jZShmZW5jZUNvbnRleHQ6IEZlbmNlQ29udGV4dCkge1xuICAgIHJldHVybiBuZXcgUHJvbWlzZTx2b2lkPihyZXNvbHZlID0+IHtcbiAgICAgIHRoaXMuYWRkSXRlbVRvUG9sbCgoKSA9PiBmZW5jZUNvbnRleHQuaXNGZW5jZVBhc3NlZCgpLCAoKSA9PiByZXNvbHZlKCkpO1xuICAgIH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBpdGVtc1RvUG9sbDogUG9sbEl0ZW1bXSA9IFtdO1xuXG4gIHBvbGxJdGVtcygpOiB2b2lkIHtcbiAgICAvLyBGaW5kIHRoZSBsYXN0IHF1ZXJ5IHRoYXQgaGFzIGZpbmlzaGVkLlxuICAgIGNvbnN0IGluZGV4ID0gbGluZWFyU2VhcmNoTGFzdFRydWUodGhpcy5pdGVtc1RvUG9sbC5tYXAoeCA9PiB4LmlzRG9uZUZuKSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPD0gaW5kZXg7ICsraSkge1xuICAgICAgY29uc3Qge3Jlc29sdmVGbn0gPSB0aGlzLml0ZW1zVG9Qb2xsW2ldO1xuICAgICAgcmVzb2x2ZUZuKCk7XG4gICAgfVxuICAgIHRoaXMuaXRlbXNUb1BvbGwgPSB0aGlzLml0ZW1zVG9Qb2xsLnNsaWNlKGluZGV4ICsgMSk7XG4gIH1cblxuICBwcml2YXRlIGFkZEl0ZW1Ub1BvbGwoaXNEb25lRm46ICgpID0+IGJvb2xlYW4sIHJlc29sdmVGbjogKCkgPT4gdm9pZCkge1xuICAgIHRoaXMuaXRlbXNUb1BvbGwucHVzaCh7aXNEb25lRm4sIHJlc29sdmVGbn0pO1xuICAgIGlmICh0aGlzLml0ZW1zVG9Qb2xsLmxlbmd0aCA+IDEpIHtcbiAgICAgIC8vIFdlIGFscmVhZHkgaGF2ZSBhIHJ1bm5pbmcgbG9vcCB0aGF0IHBvbGxzLlxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICAvLyBTdGFydCBhIG5ldyBsb29wIHRoYXQgcG9sbHMuXG4gICAgbGV0IHNjaGVkdWxlRm4gPSB1bmRlZmluZWQ7XG4gICAgaWYgKCdzZXRUaW1lb3V0Q3VzdG9tJyBpbiBlbnYoKS5wbGF0Zm9ybSkge1xuICAgICAgc2NoZWR1bGVGbiA9IGVudigpLnBsYXRmb3JtLnNldFRpbWVvdXRDdXN0b20uYmluZChlbnYoKS5wbGF0Zm9ybSk7XG4gICAgfVxuICAgIHV0aWwucmVwZWF0ZWRUcnkoKCkgPT4ge1xuICAgICAgdGhpcy5wb2xsSXRlbXMoKTtcbiAgICAgIC8vIEVuZCB0aGUgbG9vcCBpZiBubyBtb3JlIGl0ZW1zIHRvIHBvbGwuXG4gICAgICByZXR1cm4gdGhpcy5pdGVtc1RvUG9sbC5sZW5ndGggPT09IDA7XG4gICAgfSwgKCkgPT4gMCwgbnVsbCwgc2NoZWR1bGVGbik7XG4gIH1cblxuICBwcml2YXRlIGJpbmRUZXh0dXJlVG9GcmFtZUJ1ZmZlcih0ZXh0dXJlOiBXZWJHTFRleHR1cmUpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGlmICh0aGlzLmRlYnVnKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSB1bmJpbmRUZXh0dXJlVG9GcmFtZUJ1ZmZlcigpIHtcbiAgICBpZiAodGhpcy5vdXRwdXRUZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgICAgdGhpcy5nbCwgdGhpcy5vdXRwdXRUZXh0dXJlLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICAgIGlmICh0aGlzLmRlYnVnKSB7XG4gICAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBkb3dubG9hZE1hdHJpeERyaXZlcihcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICAgIGRvd25sb2FkQW5kRGVjb2RlOiAoKSA9PiBGbG9hdDMyQXJyYXkpOiBGbG9hdDMyQXJyYXkge1xuICAgIHRoaXMuYmluZFRleHR1cmVUb0ZyYW1lQnVmZmVyKHRleHR1cmUpO1xuICAgIGNvbnN0IHJlc3VsdCA9IGRvd25sb2FkQW5kRGVjb2RlKCk7XG4gICAgdGhpcy51bmJpbmRUZXh0dXJlVG9GcmFtZUJ1ZmZlcigpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihcbiAgICAgIG91dHB1dE1hdHJpeFRleHR1cmVNYXliZVBhY2tlZDogV2ViR0xUZXh0dXJlLCB3aWR0aDogbnVtYmVyLFxuICAgICAgaGVpZ2h0OiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICB3ZWJnbF91dGlsLmJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgICAgICBnbCwgb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICBpZiAodGhpcy5kZWJ1Zykge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKGdsKTtcbiAgICB9XG4gICAgdGhpcy5vdXRwdXRUZXh0dXJlID0gb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52aWV3cG9ydCgwLCAwLCB3aWR0aCwgaGVpZ2h0KSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNjaXNzb3IoMCwgMCwgd2lkdGgsIGhlaWdodCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBzZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbkRyaXZlcihcbiAgICAgIHg6IG51bWJlciwgeTogbnVtYmVyLCB3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuc2Npc3Nvcih4LCB5LCB3aWR0aCwgaGVpZ2h0KSk7XG4gIH1cblxuICBwcml2YXRlIHRocm93SWZEaXNwb3NlZCgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdBdHRlbXB0ZWQgdG8gdXNlIGRpc3Bvc2VkIEdQR1BVQ29udGV4dC4nKTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIHRocm93SWZOb1Byb2dyYW0oKSB7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ05vIEdQVSBwcm9ncmFtIGlzIGN1cnJlbnRseSBzZXQuJyk7XG4gICAgfVxuICB9XG59XG5cbnR5cGUgUG9sbEl0ZW0gPSB7XG4gIGlzRG9uZUZuOiAoKSA9PiBib29sZWFuLFxuICByZXNvbHZlRm46ICgpID0+IHZvaWRcbn07XG5cbi8qKlxuICogRmluZHMgdGhlIGluZGV4IG9mIHRoZSBsYXN0IHRydWUgZWxlbWVudCB1c2luZyBsaW5lYXIgc2VhcmNoLlxuICogTm90ZTogV2UgY2FuJ3QgZG8gYmluYXJ5IHNlYXJjaCBiZWNhdXNlIENocm9tZSBleHBlY3RzIHVzIHRvIGV4cGxpY2l0bHlcbiAqIHRlc3QgYWxsIGZlbmNlcyBiZWZvcmUgZG93bmxvYWQ6XG4gKiBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2lzc3Vlcy8xMTQ1XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBsaW5lYXJTZWFyY2hMYXN0VHJ1ZShhcnI6IEFycmF5PCgpID0+IGJvb2xlYW4+KTogbnVtYmVyIHtcbiAgbGV0IGkgPSAwO1xuICBmb3IgKDsgaSA8IGFyci5sZW5ndGg7ICsraSkge1xuICAgIGNvbnN0IGlzRG9uZSA9IGFycltpXSgpO1xuICAgIGlmICghaXNEb25lKSB7XG4gICAgICBicmVhaztcbiAgICB9XG4gIH1cbiAgcmV0dXJuIGkgLSAxO1xufVxuIl19