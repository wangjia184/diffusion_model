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
import { Fill, util } from '@tensorflow/tfjs-core';
export function fill(args) {
    const { backend, attrs } = args;
    const { shape, value, dtype } = attrs;
    const $dtype = dtype || util.inferDtype(value);
    const values = util.getArrayFromDType($dtype, util.sizeFromShape(shape));
    fillValues(values, value, $dtype);
    return backend.makeTensorInfo(shape, $dtype, values);
}
export const fillConfig = {
    kernelName: Fill,
    backendName: 'cpu',
    kernelFunc: fill
};
function fillValues(values, value, dtype) {
    if (dtype === 'string') {
        values.fill(value);
    }
    else {
        values.fill(value);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRmlsbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvRmlsbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQXVCLElBQUksRUFBK0QsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFJcEksTUFBTSxVQUFVLElBQUksQ0FBQyxJQUFpRDtJQUVwRSxNQUFNLEVBQUMsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUM5QixNQUFNLEVBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFcEMsTUFBTSxNQUFNLEdBQUcsS0FBSyxJQUFJLElBQUksQ0FBQyxVQUFVLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDL0MsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLGlCQUFpQixDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7SUFDekUsVUFBVSxDQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFFbEMsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsTUFBTSxDQUFDLENBQUM7QUFDdkQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFVBQVUsR0FBaUI7SUFDdEMsVUFBVSxFQUFFLElBQUk7SUFDaEIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLElBQTZCO0NBQzFDLENBQUM7QUFFRixTQUFTLFVBQVUsQ0FDZixNQUFrQixFQUFFLEtBQW9CLEVBQUUsS0FBZTtJQUMzRCxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7UUFDckIsTUFBbUIsQ0FBQyxJQUFJLENBQUMsS0FBZSxDQUFDLENBQUM7S0FDNUM7U0FBTTtRQUNKLE1BQXFCLENBQUMsSUFBSSxDQUFDLEtBQWUsQ0FBQyxDQUFDO0tBQzlDO0FBQ0gsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtEYXRhVHlwZSwgRGF0YVZhbHVlcywgRmlsbCwgRmlsbEF0dHJzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGZpbGwoYXJnczoge2JhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLCBhdHRyczogRmlsbEF0dHJzfSk6XG4gICAgVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7c2hhcGUsIHZhbHVlLCBkdHlwZX0gPSBhdHRycztcblxuICBjb25zdCAkZHR5cGUgPSBkdHlwZSB8fCB1dGlsLmluZmVyRHR5cGUodmFsdWUpO1xuICBjb25zdCB2YWx1ZXMgPSB1dGlsLmdldEFycmF5RnJvbURUeXBlKCRkdHlwZSwgdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKSk7XG4gIGZpbGxWYWx1ZXModmFsdWVzLCB2YWx1ZSwgJGR0eXBlKTtcblxuICByZXR1cm4gYmFja2VuZC5tYWtlVGVuc29ySW5mbyhzaGFwZSwgJGR0eXBlLCB2YWx1ZXMpO1xufVxuXG5leHBvcnQgY29uc3QgZmlsbENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBGaWxsLFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IGZpbGwgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuXG5mdW5jdGlvbiBmaWxsVmFsdWVzKFxuICAgIHZhbHVlczogRGF0YVZhbHVlcywgdmFsdWU6IHN0cmluZ3xudW1iZXIsIGR0eXBlOiBEYXRhVHlwZSk6IHZvaWQge1xuICBpZiAoZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgKHZhbHVlcyBhcyBzdHJpbmdbXSkuZmlsbCh2YWx1ZSBhcyBzdHJpbmcpO1xuICB9IGVsc2Uge1xuICAgICh2YWx1ZXMgYXMgVHlwZWRBcnJheSkuZmlsbCh2YWx1ZSBhcyBudW1iZXIpO1xuICB9XG59XG4iXX0=