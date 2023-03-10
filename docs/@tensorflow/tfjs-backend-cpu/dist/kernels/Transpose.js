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
import { Transpose } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { transposeImpl } from './Transpose_impl';
export function transpose(args) {
    const { inputs, attrs, backend } = args;
    const { x } = inputs;
    const { perm } = attrs;
    assertNotComplex(x, 'transpose');
    const xRank = x.shape.length;
    const newShape = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = x.shape[perm[i]];
    }
    const values = backend.data.get(x.dataId).values;
    const result = transposeImpl(values, x.shape, x.dtype, perm, newShape);
    const dataId = backend.write(result, newShape, x.dtype);
    return { dataId, shape: newShape, dtype: x.dtype };
}
export const transposeConfig = {
    kernelName: Transpose,
    backendName: 'cpu',
    kernelFunc: transpose
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVHJhbnNwb3NlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9UcmFuc3Bvc2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUF1QyxTQUFTLEVBQThDLE1BQU0sdUJBQXVCLENBQUM7QUFHbkksT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRTdDLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUUvQyxNQUFNLFVBQVUsU0FBUyxDQUFDLElBSXpCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUUsT0FBTyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLElBQUksRUFBQyxHQUFHLEtBQUssQ0FBQztJQUVyQixnQkFBZ0IsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFakMsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUM7SUFFN0IsTUFBTSxRQUFRLEdBQWEsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDNUMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDeEMsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDaEM7SUFFRCxNQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUMvRCxNQUFNLE1BQU0sR0FBRyxhQUFhLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFFdkUsTUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLEtBQUssQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUN4RCxPQUFPLEVBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxRQUFRLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUMsQ0FBQztBQUNuRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sZUFBZSxHQUFpQjtJQUMzQyxVQUFVLEVBQUUsU0FBUztJQUNyQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsU0FBa0M7Q0FDL0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFRyYW5zcG9zZSwgVHJhbnNwb3NlQXR0cnMsIFRyYW5zcG9zZUlucHV0cywgVHlwZWRBcnJheX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHthc3NlcnROb3RDb21wbGV4fSBmcm9tICcuLi9jcHVfdXRpbCc7XG5cbmltcG9ydCB7dHJhbnNwb3NlSW1wbH0gZnJvbSAnLi9UcmFuc3Bvc2VfaW1wbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiB0cmFuc3Bvc2UoYXJnczoge1xuICBpbnB1dHM6IFRyYW5zcG9zZUlucHV0cyxcbiAgYXR0cnM6IFRyYW5zcG9zZUF0dHJzLFxuICBiYWNrZW5kOiBNYXRoQmFja2VuZENQVVxufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBhdHRycywgYmFja2VuZH0gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtwZXJtfSA9IGF0dHJzO1xuXG4gIGFzc2VydE5vdENvbXBsZXgoeCwgJ3RyYW5zcG9zZScpO1xuXG4gIGNvbnN0IHhSYW5rID0geC5zaGFwZS5sZW5ndGg7XG5cbiAgY29uc3QgbmV3U2hhcGU6IG51bWJlcltdID0gbmV3IEFycmF5KHhSYW5rKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIG5ld1NoYXBlW2ldID0geC5zaGFwZVtwZXJtW2ldXTtcbiAgfVxuXG4gIGNvbnN0IHZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCByZXN1bHQgPSB0cmFuc3Bvc2VJbXBsKHZhbHVlcywgeC5zaGFwZSwgeC5kdHlwZSwgcGVybSwgbmV3U2hhcGUpO1xuXG4gIGNvbnN0IGRhdGFJZCA9IGJhY2tlbmQud3JpdGUocmVzdWx0LCBuZXdTaGFwZSwgeC5kdHlwZSk7XG4gIHJldHVybiB7ZGF0YUlkLCBzaGFwZTogbmV3U2hhcGUsIGR0eXBlOiB4LmR0eXBlfTtcbn1cblxuZXhwb3J0IGNvbnN0IHRyYW5zcG9zZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBUcmFuc3Bvc2UsXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogdHJhbnNwb3NlIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==