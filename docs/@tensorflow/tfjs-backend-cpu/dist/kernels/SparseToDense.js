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
import { scatterImpl } from './Scatter_impl';
export function sparseToDense(args) {
    const { inputs, backend, attrs } = args;
    const { sparseIndices, sparseValues, defaultValue } = inputs;
    const { outputShape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    const indicesBuf = backend.bufferSync(sparseIndices);
    let outBuf;
    switch (sparseValues.dtype) {
        case 'bool': {
            const updatesBuf = backend.bufferSync(sparseValues);
            const $defaultValue = Boolean(backend.data.get(defaultValue.dataId).values[0]);
            outBuf = scatterImpl(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
            break;
        }
        case 'float32': {
            const updatesBuf = backend.bufferSync(sparseValues);
            const $defaultValue = backend.data.get(defaultValue.dataId).values[0];
            outBuf = scatterImpl(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
            break;
        }
        case 'int32': {
            const updatesBuf = backend.bufferSync(sparseValues);
            const $defaultValue = backend.data.get(defaultValue.dataId).values[0];
            outBuf = scatterImpl(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
            break;
        }
        case 'string': {
            const updatesBuf = backend.bufferSync(sparseValues);
            const $defaultValue = util.decodeString(backend.data.get(defaultValue.dataId).values[0]);
            outBuf = scatterImpl(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
            break;
        }
        default:
            throw new Error(`Unsupported type ${sparseValues.dtype}`);
    }
    return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
}
export const sparseToDenseConfig = {
    kernelName: SparseToDense,
    backendName: 'cpu',
    kernelFunc: sparseToDense
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3BhcnNlVG9EZW5zZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvU3BhcnNlVG9EZW5zZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFrQyxhQUFhLEVBQXVELElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRzdKLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUUzQyxNQUFNLFVBQVUsYUFBYSxDQUFDLElBSTdCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxhQUFhLEVBQUUsWUFBWSxFQUFFLFlBQVksRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMzRCxNQUFNLEVBQUMsV0FBVyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRTVCLE1BQU0sRUFBQyxTQUFTLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFBRSxPQUFPLEVBQUUsVUFBVSxFQUFDLEdBQ3pELFlBQVksQ0FBQyxlQUFlLENBQUMsWUFBWSxFQUFFLGFBQWEsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUMzRSxNQUFNLGNBQWMsR0FBRyxLQUFLLENBQUM7SUFFN0IsTUFBTSxVQUFVLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBZ0IsYUFBYSxDQUFDLENBQUM7SUFFcEUsSUFBSSxNQUFNLENBQUM7SUFDWCxRQUFRLFlBQVksQ0FBQyxLQUFLLEVBQUU7UUFDMUIsS0FBSyxNQUFNLENBQUMsQ0FBQztZQUNYLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQWUsWUFBWSxDQUFDLENBQUM7WUFDbEUsTUFBTSxhQUFhLEdBQ2YsT0FBTyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLFlBQVksQ0FBQyxNQUFNLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUM3RCxNQUFNLEdBQUcsV0FBVyxDQUNoQixVQUFVLEVBQUUsVUFBVSxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsU0FBUyxFQUMxRCxVQUFVLEVBQUUsU0FBUyxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQUUsY0FBYyxDQUFDLENBQUM7WUFDbkUsTUFBTTtTQUNQO1FBQ0QsS0FBSyxTQUFTLENBQUMsQ0FBQztZQUNkLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQWtCLFlBQVksQ0FBQyxDQUFDO1lBQ3JFLE1BQU0sYUFBYSxHQUNmLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLFlBQVksQ0FBQyxNQUFNLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFXLENBQUM7WUFDOUQsTUFBTSxHQUFHLFdBQVcsQ0FDaEIsVUFBVSxFQUFFLFVBQVUsRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFDMUQsVUFBVSxFQUFFLFNBQVMsRUFBRSxPQUFPLEVBQUUsYUFBYSxFQUFFLGNBQWMsQ0FBQyxDQUFDO1lBQ25FLE1BQU07U0FDUDtRQUNELEtBQUssT0FBTyxDQUFDLENBQUM7WUFDWixNQUFNLFVBQVUsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFnQixZQUFZLENBQUMsQ0FBQztZQUNuRSxNQUFNLGFBQWEsR0FDZixPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxZQUFZLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBVyxDQUFDO1lBQzlELE1BQU0sR0FBRyxXQUFXLENBQ2hCLFVBQVUsRUFBRSxVQUFVLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQzFELFVBQVUsRUFBRSxTQUFTLEVBQUUsT0FBTyxFQUFFLGFBQWEsRUFBRSxjQUFjLENBQUMsQ0FBQztZQUNuRSxNQUFNO1NBQ1A7UUFDRCxLQUFLLFFBQVEsQ0FBQyxDQUFDO1lBQ2IsTUFBTSxVQUFVLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBaUIsWUFBWSxDQUFDLENBQUM7WUFDcEUsTUFBTSxhQUFhLEdBQUcsSUFBSSxDQUFDLFlBQVksQ0FDbkMsT0FBTyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDLE1BQU0sQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQWUsQ0FBQyxDQUFDO1lBQ25FLE1BQU0sR0FBRyxXQUFXLENBQ2hCLFVBQVUsRUFBRSxVQUFVLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQzFELFVBQVUsRUFBRSxTQUFTLEVBQUUsT0FBTyxFQUFFLGFBQWEsRUFBRSxjQUFjLENBQUMsQ0FBQztZQUNuRSxNQUFNO1NBQ1A7UUFDRDtZQUNFLE1BQU0sSUFBSSxLQUFLLENBQUMsb0JBQW9CLFlBQVksQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzdEO0lBQ0QsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLFdBQVcsRUFBRSxNQUFNLENBQUMsS0FBSyxFQUFFLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztBQUMxRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQWlCO0lBQy9DLFVBQVUsRUFBRSxhQUFhO0lBQ3pCLFdBQVcsRUFBRSxLQUFLO0lBQ2xCLFVBQVUsRUFBRSxhQUFzQztDQUNuRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBSYW5rLCBTcGFyc2VUb0RlbnNlLCBTcGFyc2VUb0RlbnNlQXR0cnMsIFNwYXJzZVRvRGVuc2VJbnB1dHMsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7c2NhdHRlckltcGx9IGZyb20gJy4vU2NhdHRlcl9pbXBsJztcblxuZXhwb3J0IGZ1bmN0aW9uIHNwYXJzZVRvRGVuc2UoYXJnczoge1xuICBpbnB1dHM6IFNwYXJzZVRvRGVuc2VJbnB1dHMsXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLFxuICBhdHRyczogU3BhcnNlVG9EZW5zZUF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtzcGFyc2VJbmRpY2VzLCBzcGFyc2VWYWx1ZXMsIGRlZmF1bHRWYWx1ZX0gPSBpbnB1dHM7XG4gIGNvbnN0IHtvdXRwdXRTaGFwZX0gPSBhdHRycztcblxuICBjb25zdCB7c2xpY2VSYW5rLCBudW1VcGRhdGVzLCBzbGljZVNpemUsIHN0cmlkZXMsIG91dHB1dFNpemV9ID1cbiAgICAgIGJhY2tlbmRfdXRpbC5jYWxjdWxhdGVTaGFwZXMoc3BhcnNlVmFsdWVzLCBzcGFyc2VJbmRpY2VzLCBvdXRwdXRTaGFwZSk7XG4gIGNvbnN0IHN1bUR1cGVJbmRpY2VzID0gZmFsc2U7XG5cbiAgY29uc3QgaW5kaWNlc0J1ZiA9IGJhY2tlbmQuYnVmZmVyU3luYzxSYW5rLCAnaW50MzInPihzcGFyc2VJbmRpY2VzKTtcblxuICBsZXQgb3V0QnVmO1xuICBzd2l0Y2ggKHNwYXJzZVZhbHVlcy5kdHlwZSkge1xuICAgIGNhc2UgJ2Jvb2wnOiB7XG4gICAgICBjb25zdCB1cGRhdGVzQnVmID0gYmFja2VuZC5idWZmZXJTeW5jPFJhbmssICdib29sJz4oc3BhcnNlVmFsdWVzKTtcbiAgICAgIGNvbnN0ICRkZWZhdWx0VmFsdWUgPVxuICAgICAgICAgIEJvb2xlYW4oYmFja2VuZC5kYXRhLmdldChkZWZhdWx0VmFsdWUuZGF0YUlkKS52YWx1ZXNbMF0pO1xuICAgICAgb3V0QnVmID0gc2NhdHRlckltcGwoXG4gICAgICAgICAgaW5kaWNlc0J1ZiwgdXBkYXRlc0J1Ziwgb3V0cHV0U2hhcGUsIG91dHB1dFNpemUsIHNsaWNlU2l6ZSxcbiAgICAgICAgICBudW1VcGRhdGVzLCBzbGljZVJhbmssIHN0cmlkZXMsICRkZWZhdWx0VmFsdWUsIHN1bUR1cGVJbmRpY2VzKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgICBjYXNlICdmbG9hdDMyJzoge1xuICAgICAgY29uc3QgdXBkYXRlc0J1ZiA9IGJhY2tlbmQuYnVmZmVyU3luYzxSYW5rLCAnZmxvYXQzMic+KHNwYXJzZVZhbHVlcyk7XG4gICAgICBjb25zdCAkZGVmYXVsdFZhbHVlID1cbiAgICAgICAgICBiYWNrZW5kLmRhdGEuZ2V0KGRlZmF1bHRWYWx1ZS5kYXRhSWQpLnZhbHVlc1swXSBhcyBudW1iZXI7XG4gICAgICBvdXRCdWYgPSBzY2F0dGVySW1wbChcbiAgICAgICAgICBpbmRpY2VzQnVmLCB1cGRhdGVzQnVmLCBvdXRwdXRTaGFwZSwgb3V0cHV0U2l6ZSwgc2xpY2VTaXplLFxuICAgICAgICAgIG51bVVwZGF0ZXMsIHNsaWNlUmFuaywgc3RyaWRlcywgJGRlZmF1bHRWYWx1ZSwgc3VtRHVwZUluZGljZXMpO1xuICAgICAgYnJlYWs7XG4gICAgfVxuICAgIGNhc2UgJ2ludDMyJzoge1xuICAgICAgY29uc3QgdXBkYXRlc0J1ZiA9IGJhY2tlbmQuYnVmZmVyU3luYzxSYW5rLCAnaW50MzInPihzcGFyc2VWYWx1ZXMpO1xuICAgICAgY29uc3QgJGRlZmF1bHRWYWx1ZSA9XG4gICAgICAgICAgYmFja2VuZC5kYXRhLmdldChkZWZhdWx0VmFsdWUuZGF0YUlkKS52YWx1ZXNbMF0gYXMgbnVtYmVyO1xuICAgICAgb3V0QnVmID0gc2NhdHRlckltcGwoXG4gICAgICAgICAgaW5kaWNlc0J1ZiwgdXBkYXRlc0J1Ziwgb3V0cHV0U2hhcGUsIG91dHB1dFNpemUsIHNsaWNlU2l6ZSxcbiAgICAgICAgICBudW1VcGRhdGVzLCBzbGljZVJhbmssIHN0cmlkZXMsICRkZWZhdWx0VmFsdWUsIHN1bUR1cGVJbmRpY2VzKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgICBjYXNlICdzdHJpbmcnOiB7XG4gICAgICBjb25zdCB1cGRhdGVzQnVmID0gYmFja2VuZC5idWZmZXJTeW5jPFJhbmssICdzdHJpbmcnPihzcGFyc2VWYWx1ZXMpO1xuICAgICAgY29uc3QgJGRlZmF1bHRWYWx1ZSA9IHV0aWwuZGVjb2RlU3RyaW5nKFxuICAgICAgICAgIGJhY2tlbmQuZGF0YS5nZXQoZGVmYXVsdFZhbHVlLmRhdGFJZCkudmFsdWVzWzBdIGFzIFVpbnQ4QXJyYXkpO1xuICAgICAgb3V0QnVmID0gc2NhdHRlckltcGwoXG4gICAgICAgICAgaW5kaWNlc0J1ZiwgdXBkYXRlc0J1Ziwgb3V0cHV0U2hhcGUsIG91dHB1dFNpemUsIHNsaWNlU2l6ZSxcbiAgICAgICAgICBudW1VcGRhdGVzLCBzbGljZVJhbmssIHN0cmlkZXMsICRkZWZhdWx0VmFsdWUsIHN1bUR1cGVJbmRpY2VzKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCB0eXBlICR7c3BhcnNlVmFsdWVzLmR0eXBlfWApO1xuICB9XG4gIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKG91dHB1dFNoYXBlLCBvdXRCdWYuZHR5cGUsIG91dEJ1Zi52YWx1ZXMpO1xufVxuXG5leHBvcnQgY29uc3Qgc3BhcnNlVG9EZW5zZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBTcGFyc2VUb0RlbnNlLFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IHNwYXJzZVRvRGVuc2UgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19