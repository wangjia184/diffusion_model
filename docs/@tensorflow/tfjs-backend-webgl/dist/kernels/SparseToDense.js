/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { backend_util, SparseToDense, util } from '@tensorflow/tfjs-core';
import { scatterImplCPU } from '../kernel_utils/shared';
import { ScatterProgram } from '../scatter_gpu';
import { reshape } from './Reshape';
export function sparseToDense(args) {
    const { inputs, backend, attrs } = args;
    const { sparseIndices, sparseValues, defaultValue } = inputs;
    const { outputShape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    if (sparseValues.dtype === 'string') {
        const indicesBuf = backend.bufferSync(sparseIndices);
        const updatesBuf = backend.bufferSync(sparseValues);
        const $defaultValue = util.decodeString(backend.readSync(defaultValue.dataId)[0]);
        const outBuf = scatterImplCPU(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
        return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
    }
    const program = new ScatterProgram(numUpdates, sliceRank, sparseIndices.shape.length, sparseValues.shape.length, strides, [outputSize, 1], sumDupeIndices);
    const res = backend.runWebGLProgram(program, [sparseValues, sparseIndices, defaultValue], sparseValues.dtype);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: outputShape } });
    backend.disposeIntermediateTensorInfo(res);
    return reshaped;
}
export const sparseToDenseConfig = {
    kernelName: SparseToDense,
    backendName: 'webgl',
    kernelFunc: sparseToDense
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3BhcnNlVG9EZW5zZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMva2VybmVscy9TcGFyc2VUb0RlbnNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQWtDLGFBQWEsRUFBdUQsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHN0osT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ3RELE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUU5QyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDLE1BQU0sVUFBVSxhQUFhLENBQUMsSUFJN0I7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLGFBQWEsRUFBRSxZQUFZLEVBQUUsWUFBWSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQzNELE1BQU0sRUFBQyxXQUFXLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFNUIsTUFBTSxFQUFDLFNBQVMsRUFBRSxVQUFVLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBRSxVQUFVLEVBQUMsR0FDekQsWUFBWSxDQUFDLGVBQWUsQ0FBQyxZQUFZLEVBQUUsYUFBYSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQzNFLE1BQU0sY0FBYyxHQUFHLEtBQUssQ0FBQztJQUU3QixJQUFJLFlBQVksQ0FBQyxLQUFLLEtBQUssUUFBUSxFQUFFO1FBQ25DLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQWdCLGFBQWEsQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQWlCLFlBQVksQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sYUFBYSxHQUFHLElBQUksQ0FBQyxZQUFZLENBQ25DLE9BQU8sQ0FBQyxRQUFRLENBQUMsWUFBWSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBZSxDQUFDLENBQUM7UUFDNUQsTUFBTSxNQUFNLEdBQUcsY0FBYyxDQUN6QixVQUFVLEVBQUUsVUFBVSxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsU0FBUyxFQUFFLFVBQVUsRUFDdEUsU0FBUyxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQUUsY0FBYyxDQUFDLENBQUM7UUFDdkQsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLFdBQVcsRUFBRSxNQUFNLENBQUMsS0FBSyxFQUFFLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztLQUN6RTtJQUNELE1BQU0sT0FBTyxHQUFHLElBQUksY0FBYyxDQUM5QixVQUFVLEVBQUUsU0FBUyxFQUFFLGFBQWEsQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUNqRCxZQUFZLENBQUMsS0FBSyxDQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsQ0FBQyxVQUFVLEVBQUUsQ0FBQyxDQUFDLEVBQUUsY0FBYyxDQUFDLENBQUM7SUFFekUsTUFBTSxHQUFHLEdBQUcsT0FBTyxDQUFDLGVBQWUsQ0FDL0IsT0FBTyxFQUFFLENBQUMsWUFBWSxFQUFFLGFBQWEsRUFBRSxZQUFZLENBQUMsRUFBRSxZQUFZLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFOUUsTUFBTSxRQUFRLEdBQ1YsT0FBTyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxLQUFLLEVBQUUsV0FBVyxFQUFDLEVBQUMsQ0FBQyxDQUFDO0lBRXRFLE9BQU8sQ0FBQyw2QkFBNkIsQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUMzQyxPQUFPLFFBQVEsQ0FBQztBQUNsQixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQWlCO0lBQy9DLFVBQVUsRUFBRSxhQUFhO0lBQ3pCLFdBQVcsRUFBRSxPQUFPO0lBQ3BCLFVBQVUsRUFBRSxhQUFzQztDQUNuRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBSYW5rLCBTcGFyc2VUb0RlbnNlLCBTcGFyc2VUb0RlbnNlQXR0cnMsIFNwYXJzZVRvRGVuc2VJbnB1dHMsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRXZWJHTH0gZnJvbSAnLi4vYmFja2VuZF93ZWJnbCc7XG5pbXBvcnQge3NjYXR0ZXJJbXBsQ1BVfSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvc2hhcmVkJztcbmltcG9ydCB7U2NhdHRlclByb2dyYW19IGZyb20gJy4uL3NjYXR0ZXJfZ3B1JztcblxuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuXG5leHBvcnQgZnVuY3Rpb24gc3BhcnNlVG9EZW5zZShhcmdzOiB7XG4gIGlucHV0czogU3BhcnNlVG9EZW5zZUlucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTCxcbiAgYXR0cnM6IFNwYXJzZVRvRGVuc2VBdHRyc1xufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7c3BhcnNlSW5kaWNlcywgc3BhcnNlVmFsdWVzLCBkZWZhdWx0VmFsdWV9ID0gaW5wdXRzO1xuICBjb25zdCB7b3V0cHV0U2hhcGV9ID0gYXR0cnM7XG5cbiAgY29uc3Qge3NsaWNlUmFuaywgbnVtVXBkYXRlcywgc2xpY2VTaXplLCBzdHJpZGVzLCBvdXRwdXRTaXplfSA9XG4gICAgICBiYWNrZW5kX3V0aWwuY2FsY3VsYXRlU2hhcGVzKHNwYXJzZVZhbHVlcywgc3BhcnNlSW5kaWNlcywgb3V0cHV0U2hhcGUpO1xuICBjb25zdCBzdW1EdXBlSW5kaWNlcyA9IGZhbHNlO1xuXG4gIGlmIChzcGFyc2VWYWx1ZXMuZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgY29uc3QgaW5kaWNlc0J1ZiA9IGJhY2tlbmQuYnVmZmVyU3luYzxSYW5rLCAnaW50MzInPihzcGFyc2VJbmRpY2VzKTtcbiAgICBjb25zdCB1cGRhdGVzQnVmID0gYmFja2VuZC5idWZmZXJTeW5jPFJhbmssICdzdHJpbmcnPihzcGFyc2VWYWx1ZXMpO1xuICAgIGNvbnN0ICRkZWZhdWx0VmFsdWUgPSB1dGlsLmRlY29kZVN0cmluZyhcbiAgICAgICAgYmFja2VuZC5yZWFkU3luYyhkZWZhdWx0VmFsdWUuZGF0YUlkKVswXSBhcyBVaW50OEFycmF5KTtcbiAgICBjb25zdCBvdXRCdWYgPSBzY2F0dGVySW1wbENQVShcbiAgICAgICAgaW5kaWNlc0J1ZiwgdXBkYXRlc0J1Ziwgb3V0cHV0U2hhcGUsIG91dHB1dFNpemUsIHNsaWNlU2l6ZSwgbnVtVXBkYXRlcyxcbiAgICAgICAgc2xpY2VSYW5rLCBzdHJpZGVzLCAkZGVmYXVsdFZhbHVlLCBzdW1EdXBlSW5kaWNlcyk7XG4gICAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8ob3V0cHV0U2hhcGUsIG91dEJ1Zi5kdHlwZSwgb3V0QnVmLnZhbHVlcyk7XG4gIH1cbiAgY29uc3QgcHJvZ3JhbSA9IG5ldyBTY2F0dGVyUHJvZ3JhbShcbiAgICAgIG51bVVwZGF0ZXMsIHNsaWNlUmFuaywgc3BhcnNlSW5kaWNlcy5zaGFwZS5sZW5ndGgsXG4gICAgICBzcGFyc2VWYWx1ZXMuc2hhcGUubGVuZ3RoLCBzdHJpZGVzLCBbb3V0cHV0U2l6ZSwgMV0sIHN1bUR1cGVJbmRpY2VzKTtcblxuICBjb25zdCByZXMgPSBiYWNrZW5kLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgIHByb2dyYW0sIFtzcGFyc2VWYWx1ZXMsIHNwYXJzZUluZGljZXMsIGRlZmF1bHRWYWx1ZV0sIHNwYXJzZVZhbHVlcy5kdHlwZSk7XG5cbiAgY29uc3QgcmVzaGFwZWQgPVxuICAgICAgcmVzaGFwZSh7aW5wdXRzOiB7eDogcmVzfSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogb3V0cHV0U2hhcGV9fSk7XG5cbiAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICByZXR1cm4gcmVzaGFwZWQ7XG59XG5cbmV4cG9ydCBjb25zdCBzcGFyc2VUb0RlbnNlQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFNwYXJzZVRvRGVuc2UsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBzcGFyc2VUb0RlbnNlIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==