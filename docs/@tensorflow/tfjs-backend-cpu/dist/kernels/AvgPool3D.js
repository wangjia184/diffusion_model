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
import { AvgPool3D, backend_util, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { pool3d } from '../utils/pool_utils';
export function avgPool3D(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode, dataFormat } = attrs;
    assertNotComplex(x, 'avgPool3d');
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode, dataFormat);
    const xValues = backend.data.get(x.dataId).values;
    const outBuf = pool3d(xValues, x.shape, x.dtype, util.computeStrides(x.shape), convInfo, 'avg');
    return backend.makeTensorInfo(outBuf.shape, 'float32', outBuf.values);
}
export const avgPool3DConfig = {
    kernelName: AvgPool3D,
    backendName: 'cpu',
    kernelFunc: avgPool3D
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXZnUG9vbDNELmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9BdmdQb29sM0QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFNBQVMsRUFBbUMsWUFBWSxFQUFvRCxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUd2SixPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDN0MsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLHFCQUFxQixDQUFDO0FBRTNDLE1BQU0sVUFBVSxTQUFTLENBQUMsSUFJekI7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUNuQixNQUFNLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFFLFVBQVUsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUV0RSxnQkFBZ0IsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFakMsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLGlCQUFpQixDQUMzQyxDQUFDLENBQUMsS0FBaUQsRUFBRSxVQUFVLEVBQUUsT0FBTyxFQUN4RSxDQUFDLENBQUMsZUFBZSxFQUFFLEdBQUcsRUFBRSxlQUFlLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFFekQsTUFBTSxPQUFPLEdBQUcsT0FBTyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLE1BQW9CLENBQUM7SUFDaEUsTUFBTSxNQUFNLEdBQUcsTUFBTSxDQUNqQixPQUFPLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxFQUFFLFFBQVEsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUU5RSxPQUFPLE9BQU8sQ0FBQyxjQUFjLENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxTQUFTLEVBQUUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0FBQ3hFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxlQUFlLEdBQWlCO0lBQzNDLFVBQVUsRUFBRSxTQUFTO0lBQ3JCLFdBQVcsRUFBRSxLQUFLO0lBQ2xCLFVBQVUsRUFBRSxTQUFrQztDQUMvQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0F2Z1Bvb2wzRCwgQXZnUG9vbDNEQXR0cnMsIEF2Z1Bvb2wzRElucHV0cywgYmFja2VuZF91dGlsLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7YXNzZXJ0Tm90Q29tcGxleH0gZnJvbSAnLi4vY3B1X3V0aWwnO1xuaW1wb3J0IHtwb29sM2R9IGZyb20gJy4uL3V0aWxzL3Bvb2xfdXRpbHMnO1xuXG5leHBvcnQgZnVuY3Rpb24gYXZnUG9vbDNEKGFyZ3M6IHtcbiAgaW5wdXRzOiBBdmdQb29sM0RJbnB1dHMsXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLFxuICBhdHRyczogQXZnUG9vbDNEQXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkLCBkaW1Sb3VuZGluZ01vZGUsIGRhdGFGb3JtYXR9ID0gYXR0cnM7XG5cbiAgYXNzZXJ0Tm90Q29tcGxleCh4LCAnYXZnUG9vbDNkJyk7XG5cbiAgY29uc3QgY29udkluZm8gPSBiYWNrZW5kX3V0aWwuY29tcHV0ZVBvb2wzREluZm8oXG4gICAgICB4LnNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpbHRlclNpemUsIHN0cmlkZXMsXG4gICAgICAxIC8qIGRpbGF0aW9ucyAqLywgcGFkLCBkaW1Sb3VuZGluZ01vZGUsIGRhdGFGb3JtYXQpO1xuXG4gIGNvbnN0IHhWYWx1ZXMgPSBiYWNrZW5kLmRhdGEuZ2V0KHguZGF0YUlkKS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbiAgY29uc3Qgb3V0QnVmID0gcG9vbDNkKFxuICAgICAgeFZhbHVlcywgeC5zaGFwZSwgeC5kdHlwZSwgdXRpbC5jb21wdXRlU3RyaWRlcyh4LnNoYXBlKSwgY29udkluZm8sICdhdmcnKTtcblxuICByZXR1cm4gYmFja2VuZC5tYWtlVGVuc29ySW5mbyhvdXRCdWYuc2hhcGUsICdmbG9hdDMyJywgb3V0QnVmLnZhbHVlcyk7XG59XG5cbmV4cG9ydCBjb25zdCBhdmdQb29sM0RDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQXZnUG9vbDNELFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IGF2Z1Bvb2wzRCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=