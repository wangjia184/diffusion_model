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
import { backend_util, env, util } from '@tensorflow/tfjs-core';
import * as shader_compiler from './shader_compiler';
import { createFragmentShader } from './webgl_util';
export function compileProgram(gpgpu, program, inputs, output) {
    const inputInfos = inputs.map((input, i) => {
        const shapeInfo = {
            logicalShape: input.shape,
            texShape: input.isUniform ? null : input.texData.texShape,
            isUniform: input.isUniform,
            isPacked: input.isUniform ? false : input.texData.isPacked,
            flatOffset: null
        };
        if (input.texData != null && input.texData.slice != null &&
            input.texData.slice.flatOffset > 0) {
            shapeInfo.flatOffset = input.texData.slice.flatOffset;
        }
        return { name: program.variableNames[i], shapeInfo };
    });
    const inShapeInfos = inputInfos.map(x => x.shapeInfo);
    const outShapeInfo = {
        logicalShape: output.shape,
        texShape: output.texData.texShape,
        isUniform: false,
        isPacked: output.texData.isPacked,
        flatOffset: null
    };
    const source = shader_compiler.makeShader(inputInfos, outShapeInfo, program);
    const fragmentShader = createFragmentShader(gpgpu.gl, source);
    const webGLProgram = gpgpu.createProgram(fragmentShader);
    if (!env().get('ENGINE_COMPILE_ONLY')) {
        return Object.assign({ program,
            fragmentShader,
            source,
            webGLProgram,
            inShapeInfos,
            outShapeInfo }, getUniformLocations(gpgpu, program, webGLProgram));
    }
    else {
        return {
            program,
            fragmentShader,
            source,
            webGLProgram,
            inShapeInfos,
            outShapeInfo,
            uniformLocations: null,
            customUniformLocations: null,
            infLoc: null,
            nanLoc: null,
            inShapesLocations: null,
            inTexShapesLocations: null,
            outShapeLocation: null,
            outShapeStridesLocation: null,
            outTexShapeLocation: null
        };
    }
}
export function getUniformLocations(gpgpu, program, webGLProgram) {
    const uniformLocations = {};
    const inShapesLocations = {};
    const inTexShapesLocations = {};
    const customUniformLocations = [];
    let outShapeLocation;
    let outTexShapeLocation;
    let outShapeStridesLocation;
    let infLoc = null;
    let nanLoc = null;
    // Add special uniforms (NAN, INFINITY)
    nanLoc = gpgpu.getUniformLocation(webGLProgram, 'NAN', false);
    if (env().getNumber('WEBGL_VERSION') === 1) {
        infLoc = gpgpu.getUniformLocation(webGLProgram, 'INFINITY', false);
    }
    // Add user-defined uniforms
    const shouldThrow = false;
    for (let i = 0; i < program.variableNames.length; i++) {
        const varName = program.variableNames[i];
        uniformLocations[varName] =
            gpgpu.getUniformLocation(webGLProgram, varName, shouldThrow);
        uniformLocations[`offset${varName}`] =
            gpgpu.getUniformLocation(webGLProgram, `offset${varName}`, shouldThrow);
        if (program.enableShapeUniforms) {
            inShapesLocations[`${varName}Shape`] = gpgpu.getUniformLocation(webGLProgram, `${varName}Shape`, shouldThrow);
            inTexShapesLocations[`${varName}TexShape`] = gpgpu.getUniformLocation(webGLProgram, `${varName}TexShape`, shouldThrow);
        }
    }
    if (program.enableShapeUniforms) {
        outShapeLocation =
            gpgpu.getUniformLocation(webGLProgram, 'outShape', shouldThrow);
        outShapeStridesLocation =
            gpgpu.getUniformLocation(webGLProgram, 'outShapeStrides', shouldThrow);
        outTexShapeLocation =
            gpgpu.getUniformLocation(webGLProgram, 'outTexShape', shouldThrow);
    }
    if (program.customUniforms) {
        program.customUniforms.forEach((d, i) => {
            customUniformLocations[i] =
                gpgpu.getUniformLocation(webGLProgram, d.name, shouldThrow);
        });
    }
    return {
        uniformLocations,
        customUniformLocations,
        infLoc,
        nanLoc,
        inShapesLocations,
        inTexShapesLocations,
        outShapeLocation,
        outShapeStridesLocation,
        outTexShapeLocation
    };
}
function validateBinaryAndProgram(shapeInfos, inputs) {
    if (shapeInfos.length !== inputs.length) {
        throw Error(`Binary was compiled with ${shapeInfos.length} inputs, but ` +
            `was executed with ${inputs.length} inputs`);
    }
    shapeInfos.forEach((s, i) => {
        const shapeA = s.logicalShape;
        const input = inputs[i];
        const shapeB = input.shape;
        if (!util.arraysEqual(shapeA, shapeB)) {
            throw Error(`Binary was compiled with different shapes than ` +
                `the current args. Shapes ${shapeA} and ${shapeB} must match`);
        }
        // The input is uploaded as uniform.
        if (s.isUniform && input.isUniform) {
            return;
        }
        const texShapeA = s.texShape;
        const texShapeB = input.isUniform ? null : input.texData.texShape;
        if (!util.arraysEqual(texShapeA, texShapeB)) {
            throw Error(`Binary was compiled with different texture shapes than the` +
                ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
        }
    });
}
export function runProgram(gpgpu, binary, inputs, output, customUniformValues) {
    if (!binary.program.enableShapeUniforms) {
        validateBinaryAndProgram(binary.inShapeInfos, inputs);
        validateBinaryAndProgram([binary.outShapeInfo], [output]);
    }
    const outTex = output.texData.texture;
    const outTexShape = output.texData.texShape;
    if (output.texData.isPacked) {
        gpgpu.setOutputPackedMatrixTexture(outTex.texture, outTexShape[0], outTexShape[1]);
    }
    else {
        gpgpu.setOutputMatrixTexture(outTex.texture, outTexShape[0], outTexShape[1]);
    }
    gpgpu.setProgram(binary.webGLProgram);
    // Set special uniforms (NAN, INFINITY)
    if (env().getNumber('WEBGL_VERSION') === 1) {
        if (binary.infLoc !== null) {
            gpgpu.gl.uniform1f(binary.infLoc, Infinity);
        }
    }
    if (binary.nanLoc !== null) {
        gpgpu.gl.uniform1f(binary.nanLoc, NaN);
    }
    // Set user-defined inputs
    inputs.forEach((input, i) => {
        const varName = binary.program.variableNames[i];
        const varLoc = binary.uniformLocations[varName];
        const varOffsetLoc = binary.uniformLocations[`offset${varName}`];
        const varShapeLoc = binary.inShapesLocations[`${varName}Shape`];
        const varTexShapeLoc = binary.inTexShapesLocations[`${varName}TexShape`];
        if (varShapeLoc) {
            const { uniformShape } = shader_compiler.getUniformInfoFromShape(binary.program.packedInputs, input.shape, input.texData.texShape);
            switch (uniformShape.length) {
                case 1:
                    gpgpu.gl.uniform1iv(varShapeLoc, new Int32Array(uniformShape));
                    break;
                case 2:
                    gpgpu.gl.uniform2iv(varShapeLoc, new Int32Array(uniformShape));
                    break;
                case 3:
                    gpgpu.gl.uniform3iv(varShapeLoc, new Int32Array(uniformShape));
                    break;
                case 4:
                    gpgpu.gl.uniform4iv(varShapeLoc, new Int32Array(uniformShape));
                    break;
                default:
                    break;
            }
        }
        if (varTexShapeLoc) {
            gpgpu.gl.uniform2i(varTexShapeLoc, input.texData.texShape[0], input.texData.texShape[1]);
        }
        if (varLoc == null) {
            // The compiler inferred that this variable is not used in this shader.
            return;
        }
        if (input.isUniform) {
            // Upload the values of the tensor as uniform.
            if (util.sizeFromShape(input.shape) < 2) {
                gpgpu.gl.uniform1f(varLoc, input.uniformValues[0]);
            }
            else {
                let vals = input.uniformValues;
                if (!(vals instanceof Float32Array)) {
                    vals = new Float32Array(vals);
                }
                gpgpu.gl.uniform1fv(varLoc, vals);
            }
            return;
        }
        // If the input was sliced, upload the flat offset index.
        if (input.texData.slice != null && varOffsetLoc != null) {
            gpgpu.gl.uniform1i(varOffsetLoc, input.texData.slice.flatOffset);
        }
        gpgpu.setInputMatrixTexture(input.texData.texture.texture, varLoc, i);
    });
    const outShapeLoc = binary.outShapeLocation;
    if (outShapeLoc) {
        switch (output.shape.length) {
            case 1:
                gpgpu.gl.uniform1iv(outShapeLoc, new Int32Array(output.shape));
                break;
            case 2:
                gpgpu.gl.uniform2iv(outShapeLoc, new Int32Array(output.shape));
                break;
            case 3:
                gpgpu.gl.uniform3iv(outShapeLoc, new Int32Array(output.shape));
                break;
            case 4:
                gpgpu.gl.uniform4iv(outShapeLoc, new Int32Array(output.shape));
                break;
            default:
                break;
        }
    }
    if (binary.outShapeStridesLocation) {
        const strides = util.computeStrides(output.shape);
        switch (output.shape.length) {
            case 2:
                gpgpu.gl.uniform1iv(binary.outShapeStridesLocation, new Int32Array(strides));
                break;
            case 3:
                gpgpu.gl.uniform2iv(binary.outShapeStridesLocation, new Int32Array(strides));
                break;
            case 4:
                gpgpu.gl.uniform3iv(binary.outShapeStridesLocation, new Int32Array(strides));
                break;
            default:
                break;
        }
    }
    if (binary.outTexShapeLocation) {
        gpgpu.gl.uniform2i(binary.outTexShapeLocation, output.texData.texShape[0], output.texData.texShape[1]);
    }
    if (binary.program.customUniforms && customUniformValues) {
        binary.program.customUniforms.forEach((d, i) => {
            const customLoc = binary.customUniformLocations[i];
            const customValue = customUniformValues[i];
            if (d.type === 'float') {
                gpgpu.gl.uniform1fv(customLoc, customValue);
            }
            else if (d.type === 'vec2') {
                gpgpu.gl.uniform2fv(customLoc, customValue);
            }
            else if (d.type === 'vec3') {
                gpgpu.gl.uniform3fv(customLoc, customValue);
            }
            else if (d.type === 'vec4') {
                gpgpu.gl.uniform4fv(customLoc, customValue);
            }
            else if (d.type === 'int') {
                gpgpu.gl.uniform1iv(customLoc, customValue);
            }
            else if (d.type === 'ivec2') {
                gpgpu.gl.uniform2iv(customLoc, customValue);
            }
            else if (d.type === 'ivec3') {
                gpgpu.gl.uniform3iv(customLoc, customValue);
            }
            else if (d.type === 'ivec4') {
                gpgpu.gl.uniform4iv(customLoc, customValue);
            }
            else {
                throw Error(`uniform type ${d.type} is not supported yet.`);
            }
        });
    }
    gpgpu.executeProgram();
}
export function makeShaderKey(program, inputs, output) {
    let keyInputs = '';
    inputs.concat(output).forEach(x => {
        const hasOffset = x.texData != null && x.texData.slice != null &&
            x.texData.slice.flatOffset > 0;
        // TODO: Remove the condition of !x.isUniform.
        if (program.enableShapeUniforms && !x.isUniform) {
            const xTexShape = x.texData.texShape;
            const { useSqueezeShape, uniformShape, keptDims } = shader_compiler.getUniformInfoFromShape(program.packedInputs, x.shape, xTexShape);
            let rank1 = '', rank2 = '', rank34 = '';
            if (uniformShape.length === 1 && program.packedInputs) {
                const packedTexShape = [Math.ceil(xTexShape[0] / 2), Math.ceil(xTexShape[1] / 2)];
                rank1 = `${packedTexShape[0] > 1}_${packedTexShape[1] > 1}`;
            }
            else if (uniformShape.length === 2 && !program.packedInputs) {
                rank2 = `${uniformShape[0] > 1}_${uniformShape[1] > 1}`;
            }
            else if (uniformShape.length > 2 && !program.packedInputs) {
                const strides = util.computeStrides(uniformShape);
                rank34 = `${strides[0] === xTexShape[1]}_${strides[strides.length - 1] === xTexShape[1]}`;
            }
            const xRank = x.shape.length;
            const isLogicalShapTexShapeEqual = uniformShape.length === 2 && util.arraysEqual(x.shape, xTexShape);
            const isScalar = util.sizeFromShape(x.shape) === 1;
            const broadcastDims = backend_util.getBroadcastDims(x.shape, output.shape);
            const isInOutTexShapeEqual = !program.packedInputs &&
                xRank === output.shape.length &&
                util.arraysEqual(xTexShape, output.texData.texShape);
            const isTexShapeGreaterThanOne = program.packedInputs || uniformShape.length > 2 ?
                '' :
                `${xTexShape[0] > 1}_${xTexShape[1] > 1}`;
            // These key components are needed due to shader_compiler is embedding
            // them in the shader.
            // |xRank| is used to determine the coords length. See
            // get[Packed]SamplerAtOutputCoords.
            // |isInOutTexShapeEqual| is used to determine whether going to an
            // optimization path in getSamplerAtOutputCoords.
            // |useSqueezeShape| is extracted from squeezeInputInfo of
            // getSampler[2|3|4]D/getPackedSampler3D.
            // |isScalar| is extracted from isInputScalar/isOutputScalar in
            // getPackedSamplerAtOutputCoords.
            // |broadcastDims| is extracted from get[Packed]SamplerAtOutputCoords.
            // |isLogicalShapTexShapeEqual| is used in
            // getOutput[Packed]2DCoords/get[Packed]Sampler2D.
            // |rank1| is used in getOutputPacked1DCoords.
            // |rank2| is used in getOutput2DCoords.
            // |rank34| is used in getSampler3D/getSampler4D.
            // |isTexShapeGreaterThanOne| are used in
            // getSampler[Scalar|1D|2D]/getOutput1DCoords.
            keyInputs += `${xRank}_${isInOutTexShapeEqual}_${useSqueezeShape ? keptDims : ''}_${uniformShape.length}_${isScalar}_${broadcastDims}_${isLogicalShapTexShapeEqual}_${rank1}_${rank2}_${rank34}_${isTexShapeGreaterThanOne}_${hasOffset}`;
        }
        else {
            const texShape = x.isUniform ? 'uniform' : x.texData.texShape;
            keyInputs += `${x.shape}_${texShape}_${hasOffset}`;
        }
    });
    const keyUserCode = program.userCode;
    let key = program.constructor.name;
    // Fast string concat. See https://jsperf.com/string-concatenation/14.
    key += '_' + keyInputs + '_' + keyUserCode +
        `${env().getNumber('WEBGL_VERSION')}`;
    return key;
}
export function useShapeUniforms(rank) {
    // TODO: Remove the limitaion of rank <= 4.
    return env().getBool('WEBGL_USE_SHAPES_UNIFORMS') && rank <= 4;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3BncHVfbWF0aC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvZ3BncHVfbWF0aC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFFLEdBQUcsRUFBc0IsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHbEYsT0FBTyxLQUFLLGVBQWUsTUFBTSxtQkFBbUIsQ0FBQztBQUdyRCxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSxjQUFjLENBQUM7QUErRGxELE1BQU0sVUFBVSxjQUFjLENBQzFCLEtBQW1CLEVBQUUsT0FBcUIsRUFBRSxNQUFvQixFQUNoRSxNQUFrQjtJQUNwQixNQUFNLFVBQVUsR0FBZ0IsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUN0RCxNQUFNLFNBQVMsR0FBYztZQUMzQixZQUFZLEVBQUUsS0FBSyxDQUFDLEtBQUs7WUFDekIsUUFBUSxFQUFFLEtBQUssQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxRQUFRO1lBQ3pELFNBQVMsRUFBRSxLQUFLLENBQUMsU0FBUztZQUMxQixRQUFRLEVBQUUsS0FBSyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLFFBQVE7WUFDMUQsVUFBVSxFQUFFLElBQUk7U0FDakIsQ0FBQztRQUNGLElBQUksS0FBSyxDQUFDLE9BQU8sSUFBSSxJQUFJLElBQUksS0FBSyxDQUFDLE9BQU8sQ0FBQyxLQUFLLElBQUksSUFBSTtZQUNwRCxLQUFLLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxVQUFVLEdBQUcsQ0FBQyxFQUFFO1lBQ3RDLFNBQVMsQ0FBQyxVQUFVLEdB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QUFJLG9CQUFvQixJQUN6QyxlQUFlLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxJQUFJLFlBQVksQ0FBQyxNQUFNLElBQUksUUFBUSxJQUNsRSxhQUFhLElBQUksMEJBQTBCLElBQUksS0FBSyxJQUFJLEtBQUssSUFDN0QsTUFBTSxJQUFJLHdCQUF3QixJQUFJLFNBQVMsRUFBRSxDQUFDO1NBQ3ZEO2FBQU07WUFDTCxNQUFNLFFBQVEsR0FBRyxDQUFDLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDO1lBQzlELFNBQVMsSUFBSSxHQUFHLENBQUMsQ0FBQyxLQUFLLElBQUksUUFBUSxJQUFJLFNBQVMsRUFBRSxDQUFDO1NBQ3BEO0lBQ0gsQ0FBQyxDQUFDLENBQUM7SUFDSCxNQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsUUFBUSxDQUFDO0lBQ3JDLElBQUksR0FBRyxHQUFHLE9BQU8sQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDO0lBQ25DLHNFQUFzRTtJQUN0RSxHQUFHLElBQUksR0FBRyxHQUFHLFNBQVMsR0FBRyxHQUFHLEdBQUcsV0FBVztRQUN0QyxHQUFHLEdBQUcsRUFBRSxDQUFDLFNBQVMsQ0FBQyxlQUFlLENBQUMsRUFBRSxDQUFDO0lBQzFDLE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQztBQUVELE1BQU0sVUFBVSxnQkFBZ0IsQ0FBQyxJQUFZO0lBQzNDLDJDQUEyQztJQUMzQyxPQUFPLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQywyQkFBMkIsQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLENBQUM7QUFDakUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIGVudiwgVGVuc29yLCBUeXBlZEFycmF5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dCwgR1BHUFVDb250ZXh0UHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIHNoYWRlcl9jb21waWxlciBmcm9tICcuL3NoYWRlcl9jb21waWxlcic7XG5pbXBvcnQge0lucHV0SW5mbywgU2hhcGVJbmZvLCBVbmlmb3JtVHlwZX0gZnJvbSAnLi9zaGFkZXJfY29tcGlsZXInO1xuaW1wb3J0IHtQYWNraW5nU2NoZW1lLCBUZXh0dXJlRGF0YSwgVGV4dHVyZVVzYWdlfSBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7Y3JlYXRlRnJhZ21lbnRTaGFkZXJ9IGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgdXNlckNvZGU6IHN0cmluZztcbiAgZW5hYmxlU2hhcGVVbmlmb3Jtcz86IGJvb2xlYW47XG4gIC8qKiBJZiB0cnVlLCB0aGlzIHByb2dyYW0gZXhwZWN0cyBwYWNrZWQgaW5wdXQgdGV4dHVyZXMuIERlZmF1bHRzIHRvIGZhbHNlLiAqL1xuICBwYWNrZWRJbnB1dHM/OiBib29sZWFuO1xuICAvKiogSWYgdHJ1ZSwgdGhpcyBwcm9ncmFtIHByb2R1Y2VzIGEgcGFja2VkIHRleHR1cmUuIERlZmF1bHRzIHRvIGZhbHNlLiAqL1xuICBwYWNrZWRPdXRwdXQ/OiBib29sZWFuO1xuICAvKipcbiAgICogQWZmZWN0cyB3aGF0IHR5cGUgb2YgdGV4dHVyZSB3ZSBhbGxvY2F0ZSBmb3IgdGhlIG91dHB1dC4gRGVmYXVsdHMgdG9cbiAgICogYFRleHR1cmVVc2FnZS5SRU5ERVJgLlxuICAgKi9cbiAgb3V0VGV4VXNhZ2U/OiBUZXh0dXJlVXNhZ2U7XG4gIC8qKlxuICAgKiBUaGUgdHlwZSBvZiBzY2hlbWUgdG8gdXNlIHdoZW4gcGFja2luZyB0ZXhlbHMgZm9yIHRoZSBvdXRwdXQgdmFsdWVzLlxuICAgKiBTZWUgYFBhY2tpbmdTY2hlbWVgIGZvciBkZXRhaWxzLiBEZWZhdWx0cyB0byBgUGFja2luZ1NjaGVtZS5TSEFSRURfQkFUQ0hgLlxuICAgKi9cbiAgb3V0UGFja2luZ1NjaGVtZT86IFBhY2tpbmdTY2hlbWU7XG4gIGN1c3RvbVVuaWZvcm1zPzpcbiAgICAgIEFycmF5PHtuYW1lOiBzdHJpbmc7IGFycmF5SW5kZXg/OiBudW1iZXI7IHR5cGU6IFVuaWZvcm1UeXBlO30+O1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIEdQR1BVQmluYXJ5IHtcbiAgd2ViR0xQcm9ncmFtOiBHUEdQVUNvbnRleHRQcm9ncmFtO1xuICBwcm9ncmFtOiBHUEdQVVByb2dyYW07XG4gIHVuaWZvcm1Mb2NhdGlvbnM6IHtbbmFtZTogc3RyaW5nXTogV2ViR0xVbmlmb3JtTG9jYXRpb259O1xuICBjdXN0b21Vbmlmb3JtTG9jYXRpb25zPzogV2ViR0xVbmlmb3JtTG9jYXRpb25bXTtcbiAgc291cmNlOiBzdHJpbmc7XG4gIGZyYWdtZW50U2hhZGVyOiBXZWJHTFNoYWRlcjtcbiAgaW5TaGFwZUluZm9zOiBTaGFwZUluZm9bXTtcbiAgb3V0U2hhcGVJbmZvOiBTaGFwZUluZm87XG4gIGluZkxvYzogV2ViR0xVbmlmb3JtTG9jYXRpb247XG4gIG5hbkxvYzogV2ViR0xVbmlmb3JtTG9jYXRpb247XG4gIGluU2hhcGVzTG9jYXRpb25zPzoge1tuYW1lOiBzdHJpbmddOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbn07XG4gIGluVGV4U2hhcGVzTG9jYXRpb25zPzoge1tuYW1lOiBzdHJpbmddOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbn07XG4gIG91dFNoYXBlTG9jYXRpb24/OiBXZWJHTFVuaWZvcm1Mb2NhdGlvbjtcbiAgb3V0U2hhcGVTdHJpZGVzTG9jYXRpb24/OiBXZWJHTFVuaWZvcm1Mb2NhdGlvbjtcbiAgb3V0VGV4U2hhcGVMb2NhdGlvbj86IFdlYkdMVW5pZm9ybUxvY2F0aW9uO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIEdQR1BVQmluYXJ5TG9jYXRpb25zIHtcbiAgdW5pZm9ybUxvY2F0aW9uczoge1tuYW1lOiBzdHJpbmddOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbn07XG4gIGN1c3RvbVVuaWZvcm1Mb2NhdGlvbnM/OiBXZWJHTFVuaWZvcm1Mb2NhdGlvbltdO1xuICBpbmZMb2M6IFdlYkdMVW5pZm9ybUxvY2F0aW9uO1xuICBuYW5Mb2M6IFdlYkdMVW5pZm9ybUxvY2F0aW9uO1xuICBpblNoYXBlc0xvY2F0aW9ucz86IHtbbmFtZTogc3RyaW5nXTogV2ViR0xVbmlmb3JtTG9jYXRpb259O1xuICBpblRleFNoYXBlc0xvY2F0aW9ucz86IHtbbmFtZTogc3RyaW5nXTogV2ViR0xVbmlmb3JtTG9jYXRpb259O1xuICBvdXRTaGFwZUxvY2F0aW9uPzogV2ViR0xVbmlmb3JtTG9jYXRpb247XG4gIG91dFNoYXBlU3RyaWRlc0xvY2F0aW9uPzogV2ViR0xVbmlmb3JtTG9jYXRpb247XG4gIG91dFRleFNoYXBlTG9jYXRpb24/OiBXZWJHTFVuaWZvcm1Mb2NhdGlvbjtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBUZW5zb3JEYXRhIHtcbiAgc2hhcGU6IG51bWJlcltdO1xuICB0ZXhEYXRhOiBUZXh0dXJlRGF0YTtcbiAgaXNVbmlmb3JtOiBib29sZWFuO1xuICAvLyBBdmFpbGFibGUgd2hlbiB3ZSBkZWNpZGUgdG8gdXBsb2FkIGFzIHVuaWZvcm0gaW5zdGVhZCBvZiB0ZXh0dXJlLlxuICB1bmlmb3JtVmFsdWVzPzogVHlwZWRBcnJheTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXBpbGVQcm9ncmFtPFQgZXh0ZW5kcyBUZW5zb3IsIEsgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBUZW5zb3JEYXRhW10sXG4gICAgb3V0cHV0OiBUZW5zb3JEYXRhKTogR1BHUFVCaW5hcnkge1xuICBjb25zdCBpbnB1dEluZm9zOiBJbnB1dEluZm9bXSA9IGlucHV0cy5tYXAoKGlucHV0LCBpKSA9PiB7XG4gICAgY29uc3Qgc2hhcGVJbmZvOiBTaGFwZUluZm8gPSB7XG4gICAgICBsb2dpY2FsU2hhcGU6IGlucHV0LnNoYXBlLFxuICAgICAgdGV4U2hhcGU6IGlucHV0LmlzVW5pZm9ybSA/IG51bGwgOiBpbnB1dC50ZXhEYXRhLnRleFNoYXBlLFxuICAgICAgaXNVbmlmb3JtOiBpbnB1dC5pc1VuaWZvcm0sXG4gICAgICBpc1BhY2tlZDogaW5wdXQuaXNVbmlmb3JtID8gZmFsc2UgOiBpbnB1dC50ZXhEYXRhLmlzUGFja2VkLFxuICAgICAgZmxhdE9mZnNldDogbnVsbFxuICAgIH07XG4gICAgaWYgKGlucHV0LnRleERhdGEgIT0gbnVsbCAmJiBpbnB1dC50ZXhEYXRhLnNsaWNlICE9IG51bGwgJiZcbiAgICAgICAgaW5wdXQudGV4RGF0YS5zbGljZS5mbGF0T2Zmc2V0ID4gMCkge1xuICAgICAgc2hhcGVJbmZvLmZsYXRPZmZzZXQgPSBpbnB1dC50ZXhEYXRhLnNsaWNlLmZsYXRPZmZzZXQ7XG4gICAgfVxuICAgIHJldHVybiB7bmFtZTogcHJvZ3JhbS52YXJpYWJsZU5hbWVzW2ldLCBzaGFwZUluZm99O1xuICB9KTtcbiAgY29uc3QgaW5TaGFwZUluZm9zID0gaW5wdXRJbmZvcy5tYXAoeCA9PiB4LnNoYXBlSW5mbyk7XG4gIGNvbnN0IG91dFNoYXBlSW5mbzogU2hhcGVJbmZvID0ge1xuICAgIGxvZ2ljYWxTaGFwZTogb3V0cHV0LnNoYXBlLFxuICAgIHRleFNoYXBlOiBvdXRwdXQudGV4RGF0YS50ZXhTaGFwZSxcbiAgICBpc1VuaWZvcm06IGZhbHNlLFxuICAgIGlzUGFja2VkOiBvdXRwdXQudGV4RGF0YS5pc1BhY2tlZCxcbiAgICBmbGF0T2Zmc2V0OiBudWxsXG4gIH07XG4gIGNvbnN0IHNvdXJjZSA9IHNoYWRlcl9jb21waWxlci5tYWtlU2hhZGVyKGlucHV0SW5mb3MsIG91dFNoYXBlSW5mbywgcHJvZ3JhbSk7XG4gIGNvbnN0IGZyYWdtZW50U2hhZGVyID0gY3JlYXRlRnJhZ21lbnRTaGFkZXIoZ3BncHUuZ2wsIHNvdXJjZSk7XG4gIGNvbnN0IHdlYkdMUHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0oZnJhZ21lbnRTaGFkZXIpO1xuXG4gIGlmICghZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICByZXR1cm4ge1xuICAgICAgcHJvZ3JhbSxcbiAgICAgIGZyYWdtZW50U2hhZGVyLFxuICAgICAgc291cmNlLFxuICAgICAgd2ViR0xQcm9ncmFtLFxuICAgICAgaW5TaGFwZUluZm9zLFxuICAgICAgb3V0U2hhcGVJbmZvLFxuICAgICAgLi4uZ2V0VW5pZm9ybUxvY2F0aW9ucyhncGdwdSwgcHJvZ3JhbSwgd2ViR0xQcm9ncmFtKVxuICAgIH07XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHByb2dyYW0sXG4gICAgICBmcmFnbWVudFNoYWRlcixcbiAgICAgIHNvdXJjZSxcbiAgICAgIHdlYkdMUHJvZ3JhbSxcbiAgICAgIGluU2hhcGVJbmZvcyxcbiAgICAgIG91dFNoYXBlSW5mbyxcbiAgICAgIHVuaWZvcm1Mb2NhdGlvbnM6IG51bGwsXG4gICAgICBjdXN0b21Vbmlmb3JtTG9jYXRpb25zOiBudWxsLFxuICAgICAgaW5mTG9jOiBudWxsLFxuICAgICAgbmFuTG9jOiBudWxsLFxuICAgICAgaW5TaGFwZXNMb2NhdGlvbnM6IG51bGwsXG4gICAgICBpblRleFNoYXBlc0xvY2F0aW9uczogbnVsbCxcbiAgICAgIG91dFNoYXBlTG9jYXRpb246IG51bGwsXG4gICAgICBvdXRTaGFwZVN0cmlkZXNMb2NhdGlvbjogbnVsbCxcbiAgICAgIG91dFRleFNoYXBlTG9jYXRpb246IG51bGxcbiAgICB9O1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbmlmb3JtTG9jYXRpb25zKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IEdQR1BVUHJvZ3JhbSxcbiAgICB3ZWJHTFByb2dyYW06IFdlYkdMUHJvZ3JhbSk6IEdQR1BVQmluYXJ5TG9jYXRpb25zIHtcbiAgY29uc3QgdW5pZm9ybUxvY2F0aW9uczoge1tuYW1lOiBzdHJpbmddOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbn0gPSB7fTtcbiAgY29uc3QgaW5TaGFwZXNMb2NhdGlvbnM6IHtbbmFtZTogc3RyaW5nXTogV2ViR0xVbmlmb3JtTG9jYXRpb259ID0ge307XG4gIGNvbnN0IGluVGV4U2hhcGVzTG9jYXRpb25zOiB7W25hbWU6IHN0cmluZ106IFdlYkdMVW5pZm9ybUxvY2F0aW9ufSA9IHt9O1xuICBjb25zdCBjdXN0b21Vbmlmb3JtTG9jYXRpb25zOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbltdID0gW107XG4gIGxldCBvdXRTaGFwZUxvY2F0aW9uOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbjtcbiAgbGV0IG91dFRleFNoYXBlTG9jYXRpb246IFdlYkdMVW5pZm9ybUxvY2F0aW9uO1xuICBsZXQgb3V0U2hhcGVTdHJpZGVzTG9jYXRpb246IFdlYkdMVW5pZm9ybUxvY2F0aW9uO1xuICBsZXQgaW5mTG9jOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiA9IG51bGw7XG4gIGxldCBuYW5Mb2M6IFdlYkdMVW5pZm9ybUxvY2F0aW9uID0gbnVsbDtcblxuICAvLyBBZGQgc3BlY2lhbCB1bmlmb3JtcyAoTkFOLCBJTkZJTklUWSlcbiAgbmFuTG9jID0gZ3BncHUuZ2V0VW5pZm9ybUxvY2F0aW9uKHdlYkdMUHJvZ3JhbSwgJ05BTicsIGZhbHNlKTtcbiAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpID09PSAxKSB7XG4gICAgaW5mTG9jID0gZ3BncHUuZ2V0VW5pZm9ybUxvY2F0aW9uKHdlYkdMUHJvZ3JhbSwgJ0lORklOSVRZJywgZmFsc2UpO1xuICB9XG5cbiAgLy8gQWRkIHVzZXItZGVmaW5lZCB1bmlmb3Jtc1xuICBjb25zdCBzaG91bGRUaHJvdyA9IGZhbHNlO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHByb2dyYW0udmFyaWFibGVOYW1lcy5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IHZhck5hbWUgPSBwcm9ncmFtLnZhcmlhYmxlTmFtZXNbaV07XG4gICAgdW5pZm9ybUxvY2F0aW9uc1t2YXJOYW1lXSA9XG4gICAgICAgIGdwZ3B1LmdldFVuaWZvcm1Mb2NhdGlvbih3ZWJHTFByb2dyYW0sIHZhck5hbWUsIHNob3VsZFRocm93KTtcbiAgICB1bmlmb3JtTG9jYXRpb25zW2BvZmZzZXQke3Zhck5hbWV9YF0gPVxuICAgICAgICBncGdwdS5nZXRVbmlmb3JtTG9jYXRpb24od2ViR0xQcm9ncmFtLCBgb2Zmc2V0JHt2YXJOYW1lfWAsIHNob3VsZFRocm93KTtcbiAgICBpZiAocHJvZ3JhbS5lbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgICBpblNoYXBlc0xvY2F0aW9uc1tgJHt2YXJOYW1lfVNoYXBlYF0gPSBncGdwdS5nZXRVbmlmb3JtTG9jYXRpb24oXG4gICAgICAgICAgd2ViR0xQcm9ncmFtLCBgJHt2YXJOYW1lfVNoYXBlYCwgc2hvdWxkVGhyb3cpO1xuICAgICAgaW5UZXhTaGFwZXNMb2NhdGlvbnNbYCR7dmFyTmFtZX1UZXhTaGFwZWBdID0gZ3BncHUuZ2V0VW5pZm9ybUxvY2F0aW9uKFxuICAgICAgICAgIHdlYkdMUHJvZ3JhbSwgYCR7dmFyTmFtZX1UZXhTaGFwZWAsIHNob3VsZFRocm93KTtcbiAgICB9XG4gIH1cblxuICBpZiAocHJvZ3JhbS5lbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgb3V0U2hhcGVMb2NhdGlvbiA9XG4gICAgICAgIGdwZ3B1LmdldFVuaWZvcm1Mb2NhdGlvbih3ZWJHTFByb2dyYW0sICdvdXRTaGFwZScsIHNob3VsZFRocm93KTtcbiAgICBvdXRTaGFwZVN0cmlkZXNMb2NhdGlvbiA9XG4gICAgICAgIGdwZ3B1LmdldFVuaWZvcm1Mb2NhdGlvbih3ZWJHTFByb2dyYW0sICdvdXRTaGFwZVN0cmlkZXMnLCBzaG91bGRUaHJvdyk7XG4gICAgb3V0VGV4U2hhcGVMb2NhdGlvbiA9XG4gICAgICAgIGdwZ3B1LmdldFVuaWZvcm1Mb2NhdGlvbih3ZWJHTFByb2dyYW0sICdvdXRUZXhTaGFwZScsIHNob3VsZFRocm93KTtcbiAgfVxuXG4gIGlmIChwcm9ncmFtLmN1c3RvbVVuaWZvcm1zKSB7XG4gICAgcHJvZ3JhbS5jdXN0b21Vbmlmb3Jtcy5mb3JFYWNoKChkLCBpKSA9PiB7XG4gICAgICBjdXN0b21Vbmlmb3JtTG9jYXRpb25zW2ldID1cbiAgICAgICAgICBncGdwdS5nZXRVbmlmb3JtTG9jYXRpb24od2ViR0xQcm9ncmFtLCBkLm5hbWUsIHNob3VsZFRocm93KTtcbiAgICB9KTtcbiAgfVxuXG4gIHJldHVybiB7XG4gICAgdW5pZm9ybUxvY2F0aW9ucyxcbiAgICBjdXN0b21Vbmlmb3JtTG9jYXRpb25zLFxuICAgIGluZkxvYyxcbiAgICBuYW5Mb2MsXG4gICAgaW5TaGFwZXNMb2NhdGlvbnMsXG4gICAgaW5UZXhTaGFwZXNMb2NhdGlvbnMsXG4gICAgb3V0U2hhcGVMb2NhdGlvbixcbiAgICBvdXRTaGFwZVN0cmlkZXNMb2NhdGlvbixcbiAgICBvdXRUZXhTaGFwZUxvY2F0aW9uXG4gIH07XG59XG5cbmZ1bmN0aW9uIHZhbGlkYXRlQmluYXJ5QW5kUHJvZ3JhbShcbiAgICBzaGFwZUluZm9zOiBTaGFwZUluZm9bXSwgaW5wdXRzOiBUZW5zb3JEYXRhW10pIHtcbiAgaWYgKHNoYXBlSW5mb3MubGVuZ3RoICE9PSBpbnB1dHMubGVuZ3RoKSB7XG4gICAgdGhyb3cgRXJyb3IoXG4gICAgICAgIGBCaW5hcnkgd2FzIGNvbXBpbGVkIHdpdGggJHtzaGFwZUluZm9zLmxlbmd0aH0gaW5wdXRzLCBidXQgYCArXG4gICAgICAgIGB3YXMgZXhlY3V0ZWQgd2l0aCAke2lucHV0cy5sZW5ndGh9IGlucHV0c2ApO1xuICB9XG5cbiAgc2hhcGVJbmZvcy5mb3JFYWNoKChzLCBpKSA9PiB7XG4gICAgY29uc3Qgc2hhcGVBID0gcy5sb2dpY2FsU2hhcGU7XG4gICAgY29uc3QgaW5wdXQgPSBpbnB1dHNbaV07XG4gICAgY29uc3Qgc2hhcGVCID0gaW5wdXQuc2hhcGU7XG5cbiAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwoc2hhcGVBLCBzaGFwZUIpKSB7XG4gICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICBgQmluYXJ5IHdhcyBjb21waWxlZCB3aXRoIGRpZmZlcmVudCBzaGFwZXMgdGhhbiBgICtcbiAgICAgICAgICBgdGhlIGN1cnJlbnQgYXJncy4gU2hhcGVzICR7c2hhcGVBfSBhbmQgJHtzaGFwZUJ9IG11c3QgbWF0Y2hgKTtcbiAgICB9XG4gICAgLy8gVGhlIGlucHV0IGlzIHVwbG9hZGVkIGFzIHVuaWZvcm0uXG4gICAgaWYgKHMuaXNVbmlmb3JtICYmIGlucHV0LmlzVW5pZm9ybSkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHRleFNoYXBlQSA9IHMudGV4U2hhcGU7XG4gICAgY29uc3QgdGV4U2hhcGVCID0gaW5wdXQuaXNVbmlmb3JtID8gbnVsbCA6IGlucHV0LnRleERhdGEudGV4U2hhcGU7XG4gICAgaWYgKCF1dGlsLmFycmF5c0VxdWFsKHRleFNoYXBlQSwgdGV4U2hhcGVCKSkge1xuICAgICAgdGhyb3cgRXJyb3IoXG4gICAgICAgICAgYEJpbmFyeSB3YXMgY29tcGlsZWQgd2l0aCBkaWZmZXJlbnQgdGV4dHVyZSBzaGFwZXMgdGhhbiB0aGVgICtcbiAgICAgICAgICBgIGN1cnJlbnQgYXJncy4gU2hhcGUgJHt0ZXhTaGFwZUF9IGFuZCAke3RleFNoYXBlQn0gbXVzdCBtYXRjaGApO1xuICAgIH1cbiAgfSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBydW5Qcm9ncmFtPFQgZXh0ZW5kcyBUZW5zb3IsIEsgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIGJpbmFyeTogR1BHUFVCaW5hcnksIGlucHV0czogVGVuc29yRGF0YVtdLFxuICAgIG91dHB1dDogVGVuc29yRGF0YSwgY3VzdG9tVW5pZm9ybVZhbHVlcz86IG51bWJlcltdW10pOiB2b2lkIHtcbiAgaWYgKCFiaW5hcnkucHJvZ3JhbS5lbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKGJpbmFyeS5pblNoYXBlSW5mb3MsIGlucHV0cyk7XG4gICAgdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKFtiaW5hcnkub3V0U2hhcGVJbmZvXSwgW291dHB1dF0pO1xuICB9XG5cbiAgY29uc3Qgb3V0VGV4ID0gb3V0cHV0LnRleERhdGEudGV4dHVyZTtcbiAgY29uc3Qgb3V0VGV4U2hhcGUgPSBvdXRwdXQudGV4RGF0YS50ZXhTaGFwZTtcbiAgaWYgKG91dHB1dC50ZXhEYXRhLmlzUGFja2VkKSB7XG4gICAgZ3BncHUuc2V0T3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgICAgb3V0VGV4LnRleHR1cmUsIG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSk7XG4gIH0gZWxzZSB7XG4gICAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgICAgb3V0VGV4LnRleHR1cmUsIG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSk7XG4gIH1cbiAgZ3BncHUuc2V0UHJvZ3JhbShiaW5hcnkud2ViR0xQcm9ncmFtKTtcblxuICAvLyBTZXQgc3BlY2lhbCB1bmlmb3JtcyAoTkFOLCBJTkZJTklUWSlcbiAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpID09PSAxKSB7XG4gICAgaWYgKGJpbmFyeS5pbmZMb2MgIT09IG51bGwpIHtcbiAgICAgIGdwZ3B1LmdsLnVuaWZvcm0xZihiaW5hcnkuaW5mTG9jLCBJbmZpbml0eSk7XG4gICAgfVxuICB9XG4gIGlmIChiaW5hcnkubmFuTG9jICE9PSBudWxsKSB7XG4gICAgZ3BncHUuZ2wudW5pZm9ybTFmKGJpbmFyeS5uYW5Mb2MsIE5hTik7XG4gIH1cblxuICAvLyBTZXQgdXNlci1kZWZpbmVkIGlucHV0c1xuICBpbnB1dHMuZm9yRWFjaCgoaW5wdXQsIGkpID0+IHtcbiAgICBjb25zdCB2YXJOYW1lID0gYmluYXJ5LnByb2dyYW0udmFyaWFibGVOYW1lc1tpXTtcbiAgICBjb25zdCB2YXJMb2MgPSBiaW5hcnkudW5pZm9ybUxvY2F0aW9uc1t2YXJOYW1lXTtcbiAgICBjb25zdCB2YXJPZmZzZXRMb2MgPSBiaW5hcnkudW5pZm9ybUxvY2F0aW9uc1tgb2Zmc2V0JHt2YXJOYW1lfWBdO1xuICAgIGNvbnN0IHZhclNoYXBlTG9jID0gYmluYXJ5LmluU2hhcGVzTG9jYXRpb25zW2Ake3Zhck5hbWV9U2hhcGVgXTtcbiAgICBjb25zdCB2YXJUZXhTaGFwZUxvYyA9IGJpbmFyeS5pblRleFNoYXBlc0xvY2F0aW9uc1tgJHt2YXJOYW1lfVRleFNoYXBlYF07XG5cbiAgICBpZiAodmFyU2hhcGVMb2MpIHtcbiAgICAgIGNvbnN0IHt1bmlmb3JtU2hhcGV9ID0gc2hhZGVyX2NvbXBpbGVyLmdldFVuaWZvcm1JbmZvRnJvbVNoYXBlKFxuICAgICAgICAgIGJpbmFyeS5wcm9ncmFtLnBhY2tlZElucHV0cywgaW5wdXQuc2hhcGUsIGlucHV0LnRleERhdGEudGV4U2hhcGUpO1xuICAgICAgc3dpdGNoICh1bmlmb3JtU2hhcGUubGVuZ3RoKSB7XG4gICAgICAgIGNhc2UgMTpcbiAgICAgICAgICBncGdwdS5nbC51bmlmb3JtMWl2KHZhclNoYXBlTG9jLCBuZXcgSW50MzJBcnJheSh1bmlmb3JtU2hhcGUpKTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgY2FzZSAyOlxuICAgICAgICAgIGdwZ3B1LmdsLnVuaWZvcm0yaXYodmFyU2hhcGVMb2MsIG5ldyBJbnQzMkFycmF5KHVuaWZvcm1TaGFwZSkpO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICBjYXNlIDM6XG4gICAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTNpdih2YXJTaGFwZUxvYywgbmV3IEludDMyQXJyYXkodW5pZm9ybVNoYXBlKSk7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgNDpcbiAgICAgICAgICBncGdwdS5nbC51bmlmb3JtNGl2KHZhclNoYXBlTG9jLCBuZXcgSW50MzJBcnJheSh1bmlmb3JtU2hhcGUpKTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgZGVmYXVsdDpcbiAgICAgICAgICBicmVhaztcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKHZhclRleFNoYXBlTG9jKSB7XG4gICAgICBncGdwdS5nbC51bmlmb3JtMmkoXG4gICAgICAgICAgdmFyVGV4U2hhcGVMb2MsIGlucHV0LnRleERhdGEudGV4U2hhcGVbMF0sIGlucHV0LnRleERhdGEudGV4U2hhcGVbMV0pO1xuICAgIH1cblxuICAgIGlmICh2YXJMb2MgPT0gbnVsbCkge1xuICAgICAgLy8gVGhlIGNvbXBpbGVyIGluZmVycmVkIHRoYXQgdGhpcyB2YXJpYWJsZSBpcyBub3QgdXNlZCBpbiB0aGlzIHNoYWRlci5cbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBpZiAoaW5wdXQuaXNVbmlmb3JtKSB7XG4gICAgICAvLyBVcGxvYWQgdGhlIHZhbHVlcyBvZiB0aGUgdGVuc29yIGFzIHVuaWZvcm0uXG4gICAgICBpZiAodXRpbC5zaXplRnJvbVNoYXBlKGlucHV0LnNoYXBlKSA8IDIpIHtcbiAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTFmKHZhckxvYywgaW5wdXQudW5pZm9ybVZhbHVlc1swXSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBsZXQgdmFscyA9IGlucHV0LnVuaWZvcm1WYWx1ZXM7XG4gICAgICAgIGlmICghKHZhbHMgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKSB7XG4gICAgICAgICAgdmFscyA9IG5ldyBGbG9hdDMyQXJyYXkodmFscyk7XG4gICAgICAgIH1cbiAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTFmdih2YXJMb2MsIHZhbHMpO1xuICAgICAgfVxuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIElmIHRoZSBpbnB1dCB3YXMgc2xpY2VkLCB1cGxvYWQgdGhlIGZsYXQgb2Zmc2V0IGluZGV4LlxuICAgIGlmIChpbnB1dC50ZXhEYXRhLnNsaWNlICE9IG51bGwgJiYgdmFyT2Zmc2V0TG9jICE9IG51bGwpIHtcbiAgICAgIGdwZ3B1LmdsLnVuaWZvcm0xaSh2YXJPZmZzZXRMb2MsIGlucHV0LnRleERhdGEuc2xpY2UuZmxhdE9mZnNldCk7XG4gICAgfVxuXG4gICAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGlucHV0LnRleERhdGEudGV4dHVyZS50ZXh0dXJlLCB2YXJMb2MsIGkpO1xuICB9KTtcblxuICBjb25zdCBvdXRTaGFwZUxvYyA9IGJpbmFyeS5vdXRTaGFwZUxvY2F0aW9uO1xuICBpZiAob3V0U2hhcGVMb2MpIHtcbiAgICBzd2l0Y2ggKG91dHB1dC5zaGFwZS5sZW5ndGgpIHtcbiAgICAgIGNhc2UgMTpcbiAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTFpdihvdXRTaGFwZUxvYywgbmV3IEludDMyQXJyYXkob3V0cHV0LnNoYXBlKSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgY2FzZSAyOlxuICAgICAgICBncGdwdS5nbC51bmlmb3JtMml2KG91dFNoYXBlTG9jLCBuZXcgSW50MzJBcnJheShvdXRwdXQuc2hhcGUpKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICBjYXNlIDM6XG4gICAgICAgIGdwZ3B1LmdsLnVuaWZvcm0zaXYob3V0U2hhcGVMb2MsIG5ldyBJbnQzMkFycmF5KG91dHB1dC5zaGFwZSkpO1xuICAgICAgICBicmVhaztcbiAgICAgIGNhc2UgNDpcbiAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTRpdihvdXRTaGFwZUxvYywgbmV3IEludDMyQXJyYXkob3V0cHV0LnNoYXBlKSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgYnJlYWs7XG4gICAgfVxuICB9XG4gIGlmIChiaW5hcnkub3V0U2hhcGVTdHJpZGVzTG9jYXRpb24pIHtcbiAgICBjb25zdCBzdHJpZGVzID0gdXRpbC5jb21wdXRlU3RyaWRlcyhvdXRwdXQuc2hhcGUpO1xuICAgIHN3aXRjaCAob3V0cHV0LnNoYXBlLmxlbmd0aCkge1xuICAgICAgY2FzZSAyOlxuICAgICAgICBncGdwdS5nbC51bmlmb3JtMWl2KFxuICAgICAgICAgICAgYmluYXJ5Lm91dFNoYXBlU3RyaWRlc0xvY2F0aW9uLCBuZXcgSW50MzJBcnJheShzdHJpZGVzKSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgY2FzZSAzOlxuICAgICAgICBncGdwdS5nbC51bmlmb3JtMml2KFxuICAgICAgICAgICAgYmluYXJ5Lm91dFNoYXBlU3RyaWRlc0xvY2F0aW9uLCBuZXcgSW50MzJBcnJheShzdHJpZGVzKSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgY2FzZSA0OlxuICAgICAgICBncGdwdS5nbC51bmlmb3JtM2l2KFxuICAgICAgICAgICAgYmluYXJ5Lm91dFNoYXBlU3RyaWRlc0xvY2F0aW9uLCBuZXcgSW50MzJBcnJheShzdHJpZGVzKSk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgYnJlYWs7XG4gICAgfVxuICB9XG4gIGlmIChiaW5hcnkub3V0VGV4U2hhcGVMb2NhdGlvbikge1xuICAgIGdwZ3B1LmdsLnVuaWZvcm0yaShcbiAgICAgICAgYmluYXJ5Lm91dFRleFNoYXBlTG9jYXRpb24sIG91dHB1dC50ZXhEYXRhLnRleFNoYXBlWzBdLFxuICAgICAgICBvdXRwdXQudGV4RGF0YS50ZXhTaGFwZVsxXSk7XG4gIH1cblxuICBpZiAoYmluYXJ5LnByb2dyYW0uY3VzdG9tVW5pZm9ybXMgJiYgY3VzdG9tVW5pZm9ybVZhbHVlcykge1xuICAgIGJpbmFyeS5wcm9ncmFtLmN1c3RvbVVuaWZvcm1zLmZvckVhY2goKGQsIGkpID0+IHtcbiAgICAgIGNvbnN0IGN1c3RvbUxvYyA9IGJpbmFyeS5jdXN0b21Vbmlmb3JtTG9jYXRpb25zW2ldO1xuICAgICAgY29uc3QgY3VzdG9tVmFsdWUgPSBjdXN0b21Vbmlmb3JtVmFsdWVzW2ldO1xuICAgICAgaWYgKGQudHlwZSA9PT0gJ2Zsb2F0Jykge1xuICAgICAgICBncGdwdS5nbC51bmlmb3JtMWZ2KGN1c3RvbUxvYywgY3VzdG9tVmFsdWUpO1xuICAgICAgfSBlbHNlIGlmIChkLnR5cGUgPT09ICd2ZWMyJykge1xuICAgICAgICBncGdwdS5nbC51bmlmb3JtMmZ2KGN1c3RvbUxvYywgY3VzdG9tVmFsdWUpO1xuICAgICAgfSBlbHNlIGlmIChkLnR5cGUgPT09ICd2ZWMzJykge1xuICAgICAgICBncGdwdS5nbC51bmlmb3JtM2Z2KGN1c3RvbUxvYywgY3VzdG9tVmFsdWUpO1xuICAgICAgfSBlbHNlIGlmIChkLnR5cGUgPT09ICd2ZWM0Jykge1xuICAgICAgICBncGdwdS5nbC51bmlmb3JtNGZ2KGN1c3RvbUxvYywgY3VzdG9tVmFsdWUpO1xuICAgICAgfSBlbHNlIGlmIChkLnR5cGUgPT09ICdpbnQnKSB7XG4gICAgICAgIGdwZ3B1LmdsLnVuaWZvcm0xaXYoY3VzdG9tTG9jLCBjdXN0b21WYWx1ZSk7XG4gICAgICB9IGVsc2UgaWYgKGQudHlwZSA9PT0gJ2l2ZWMyJykge1xuICAgICAgICBncGdwdS5nbC51bmlmb3JtMml2KGN1c3RvbUxvYywgY3VzdG9tVmFsdWUpO1xuICAgICAgfSBlbHNlIGlmIChkLnR5cGUgPT09ICdpdmVjMycpIHtcbiAgICAgICAgZ3BncHUuZ2wudW5pZm9ybTNpdihjdXN0b21Mb2MsIGN1c3RvbVZhbHVlKTtcbiAgICAgIH0gZWxzZSBpZiAoZC50eXBlID09PSAnaXZlYzQnKSB7XG4gICAgICAgIGdwZ3B1LmdsLnVuaWZvcm00aXYoY3VzdG9tTG9jLCBjdXN0b21WYWx1ZSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBFcnJvcihgdW5pZm9ybSB0eXBlICR7ZC50eXBlfSBpcyBub3Qgc3VwcG9ydGVkIHlldC5gKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZVNoYWRlcktleShcbiAgICBwcm9ncmFtOiBHUEdQVVByb2dyYW0sIGlucHV0czogVGVuc29yRGF0YVtdLCBvdXRwdXQ6IFRlbnNvckRhdGEpOiBzdHJpbmcge1xuICBsZXQga2V5SW5wdXRzID0gJyc7XG4gIGlucHV0cy5jb25jYXQob3V0cHV0KS5mb3JFYWNoKHggPT4ge1xuICAgIGNvbnN0IGhhc09mZnNldCA9IHgudGV4RGF0YSAhPSBudWxsICYmIHgudGV4RGF0YS5zbGljZSAhPSBudWxsICYmXG4gICAgICAgIHgudGV4RGF0YS5zbGljZS5mbGF0T2Zmc2V0ID4gMDtcbiAgICAvLyBUT0RPOiBSZW1vdmUgdGhlIGNvbmRpdGlvbiBvZiAheC5pc1VuaWZvcm0uXG4gICAgaWYgKHByb2dyYW0uZW5hYmxlU2hhcGVVbmlmb3JtcyAmJiAheC5pc1VuaWZvcm0pIHtcbiAgICAgIGNvbnN0IHhUZXhTaGFwZSA9IHgudGV4RGF0YS50ZXhTaGFwZTtcbiAgICAgIGNvbnN0IHt1c2VTcXVlZXplU2hhcGUsIHVuaWZvcm1TaGFwZSwga2VwdERpbXN9ID1cbiAgICAgICAgICBzaGFkZXJfY29tcGlsZXIuZ2V0VW5pZm9ybUluZm9Gcm9tU2hhcGUoXG4gICAgICAgICAgICAgIHByb2dyYW0ucGFja2VkSW5wdXRzLCB4LnNoYXBlLCB4VGV4U2hhcGUpO1xuICAgICAgbGV0IHJhbmsxID0gJycsIHJhbmsyID0gJycsIHJhbmszNCA9ICcnO1xuICAgICAgaWYgKHVuaWZvcm1TaGFwZS5sZW5ndGggPT09IDEgJiYgcHJvZ3JhbS5wYWNrZWRJbnB1dHMpIHtcbiAgICAgICAgY29uc3QgcGFja2VkVGV4U2hhcGUgPVxuICAgICAgICAgICAgW01hdGguY2VpbCh4VGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHhUZXhTaGFwZVsxXSAvIDIpXTtcbiAgICAgICAgcmFuazEgPSBgJHtwYWNrZWRUZXhTaGFwZVswXSA+IDF9XyR7cGFja2VkVGV4U2hhcGVbMV0gPiAxfWA7XG4gICAgICB9IGVsc2UgaWYgKHVuaWZvcm1TaGFwZS5sZW5ndGggPT09IDIgJiYgIXByb2dyYW0ucGFja2VkSW5wdXRzKSB7XG4gICAgICAgIHJhbmsyID0gYCR7dW5pZm9ybVNoYXBlWzBdID4gMX1fJHt1bmlmb3JtU2hhcGVbMV0gPiAxfWA7XG4gICAgICB9IGVsc2UgaWYgKHVuaWZvcm1TaGFwZS5sZW5ndGggPiAyICYmICFwcm9ncmFtLnBhY2tlZElucHV0cykge1xuICAgICAgICBjb25zdCBzdHJpZGVzID0gdXRpbC5jb21wdXRlU3RyaWRlcyh1bmlmb3JtU2hhcGUpO1xuICAgICAgICByYW5rMzQgPSBgJHtzdHJpZGVzWzBdID09PSB4VGV4U2hhcGVbMV19XyR7XG4gICAgICAgICAgICBzdHJpZGVzW3N0cmlkZXMubGVuZ3RoIC0gMV0gPT09IHhUZXhTaGFwZVsxXX1gO1xuICAgICAgfVxuICAgICAgY29uc3QgeFJhbmsgPSB4LnNoYXBlLmxlbmd0aDtcbiAgICAgIGNvbnN0IGlzTG9naWNhbFNoYXBUZXhTaGFwZUVxdWFsID1cbiAgICAgICAgICB1bmlmb3JtU2hhcGUubGVuZ3RoID09PSAyICYmIHV0aWwuYXJyYXlzRXF1YWwoeC5zaGFwZSwgeFRleFNoYXBlKTtcbiAgICAgIGNvbnN0IGlzU2NhbGFyID0gdXRpbC5zaXplRnJvbVNoYXBlKHguc2hhcGUpID09PSAxO1xuICAgICAgY29uc3QgYnJvYWRjYXN0RGltcyA9XG4gICAgICAgICAgYmFja2VuZF91dGlsLmdldEJyb2FkY2FzdERpbXMoeC5zaGFwZSwgb3V0cHV0LnNoYXBlKTtcbiAgICAgIGNvbnN0IGlzSW5PdXRUZXhTaGFwZUVxdWFsID0gIXByb2dyYW0ucGFja2VkSW5wdXRzICYmXG4gICAgICAgICAgeFJhbmsgPT09IG91dHB1dC5zaGFwZS5sZW5ndGggJiZcbiAgICAgICAgICB1dGlsLmFycmF5c0VxdWFsKHhUZXhTaGFwZSwgb3V0cHV0LnRleERhdGEudGV4U2hhcGUpO1xuICAgICAgY29uc3QgaXNUZXhTaGFwZUdyZWF0ZXJUaGFuT25lID1cbiAgICAgICAgICBwcm9ncmFtLnBhY2tlZElucHV0cyB8fCB1bmlmb3JtU2hhcGUubGVuZ3RoID4gMiA/XG4gICAgICAgICAgJycgOlxuICAgICAgICAgIGAke3hUZXhTaGFwZVswXSA+IDF9XyR7eFRleFNoYXBlWzFdID4gMX1gO1xuICAgICAgLy8gVGhlc2Uga2V5IGNvbXBvbmVudHMgYXJlIG5lZWRlZCBkdWUgdG8gc2hhZGVyX2NvbXBpbGVyIGlzIGVtYmVkZGluZ1xuICAgICAgLy8gdGhlbSBpbiB0aGUgc2hhZGVyLlxuICAgICAgLy8gfHhSYW5rfCBpcyB1c2VkIHRvIGRldGVybWluZSB0aGUgY29vcmRzIGxlbmd0aC4gU2VlXG4gICAgICAvLyBnZXRbUGFja2VkXVNhbXBsZXJBdE91dHB1dENvb3Jkcy5cbiAgICAgIC8vIHxpc0luT3V0VGV4U2hhcGVFcXVhbHwgaXMgdXNlZCB0byBkZXRlcm1pbmUgd2hldGhlciBnb2luZyB0byBhblxuICAgICAgLy8gb3B0aW1pemF0aW9uIHBhdGggaW4gZ2V0U2FtcGxlckF0T3V0cHV0Q29vcmRzLlxuICAgICAgLy8gfHVzZVNxdWVlemVTaGFwZXwgaXMgZXh0cmFjdGVkIGZyb20gc3F1ZWV6ZUlucHV0SW5mbyBvZlxuICAgICAgLy8gZ2V0U2FtcGxlclsyfDN8NF1EL2dldFBhY2tlZFNhbXBsZXIzRC5cbiAgICAgIC8vIHxpc1NjYWxhcnwgaXMgZXh0cmFjdGVkIGZyb20gaXNJbnB1dFNjYWxhci9pc091dHB1dFNjYWxhciBpblxuICAgICAgLy8gZ2V0UGFja2VkU2FtcGxlckF0T3V0cHV0Q29vcmRzLlxuICAgICAgLy8gfGJyb2FkY2FzdERpbXN8IGlzIGV4dHJhY3RlZCBmcm9tIGdldFtQYWNrZWRdU2FtcGxlckF0T3V0cHV0Q29vcmRzLlxuICAgICAgLy8gfGlzTG9naWNhbFNoYXBUZXhTaGFwZUVxdWFsfCBpcyB1c2VkIGluXG4gICAgICAvLyBnZXRPdXRwdXRbUGFja2VkXTJEQ29vcmRzL2dldFtQYWNrZWRdU2FtcGxlcjJELlxuICAgICAgLy8gfHJhbmsxfCBpcyB1c2VkIGluIGdldE91dHB1dFBhY2tlZDFEQ29vcmRzLlxuICAgICAgLy8gfHJhbmsyfCBpcyB1c2VkIGluIGdldE91dHB1dDJEQ29vcmRzLlxuICAgICAgLy8gfHJhbmszNHwgaXMgdXNlZCBpbiBnZXRTYW1wbGVyM0QvZ2V0U2FtcGxlcjRELlxuICAgICAgLy8gfGlzVGV4U2hhcGVHcmVhdGVyVGhhbk9uZXwgYXJlIHVzZWQgaW5cbiAgICAgIC8vIGdldFNhbXBsZXJbU2NhbGFyfDFEfDJEXS9nZXRPdXRwdXQxRENvb3Jkcy5cbiAgICAgIGtleUlucHV0cyArPSBgJHt4UmFua31fJHtpc0luT3V0VGV4U2hhcGVFcXVhbH1fJHtcbiAgICAgICAgICB1c2VTcXVlZXplU2hhcGUgPyBrZXB0RGltcyA6ICcnfV8ke3VuaWZvcm1TaGFwZS5sZW5ndGh9XyR7aXNTY2FsYXJ9XyR7XG4gICAgICAgICAgYnJvYWRjYXN0RGltc31fJHtpc0xvZ2ljYWxTaGFwVGV4U2hhcGVFcXVhbH1fJHtyYW5rMX1fJHtyYW5rMn1fJHtcbiAgICAgICAgICByYW5rMzR9XyR7aXNUZXhTaGFwZUdyZWF0ZXJUaGFuT25lfV8ke2hhc09mZnNldH1gO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCB0ZXhTaGFwZSA9IHguaXNVbmlmb3JtID8gJ3VuaWZvcm0nIDogeC50ZXhEYXRhLnRleFNoYXBlO1xuICAgICAga2V5SW5wdXRzICs9IGAke3guc2hhcGV9XyR7dGV4U2hhcGV9XyR7aGFzT2Zmc2V0fWA7XG4gICAgfVxuICB9KTtcbiAgY29uc3Qga2V5VXNlckNvZGUgPSBwcm9ncmFtLnVzZXJDb2RlO1xuICBsZXQga2V5ID0gcHJvZ3JhbS5jb25zdHJ1Y3Rvci5uYW1lO1xuICAvLyBGYXN0IHN0cmluZyBjb25jYXQuIFNlZSBodHRwczovL2pzcGVyZi5jb20vc3RyaW5nLWNvbmNhdGVuYXRpb24vMTQuXG4gIGtleSArPSAnXycgKyBrZXlJbnB1dHMgKyAnXycgKyBrZXlVc2VyQ29kZSArXG4gICAgICBgJHtlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKX1gO1xuICByZXR1cm4ga2V5O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXNlU2hhcGVVbmlmb3JtcyhyYW5rOiBudW1iZXIpIHtcbiAgLy8gVE9ETzogUmVtb3ZlIHRoZSBsaW1pdGFpb24gb2YgcmFuayA8PSA0LlxuICByZXR1cm4gZW52KCkuZ2V0Qm9vbCgnV0VCR0xfVVNFX1NIQVBFU19VTklGT1JNUycpICYmIHJhbmsgPD0gNDtcbn1cbiJdfQ==