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
import { backend_util, buffer, Slice, slice_util, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
export function sliceImpl(vals, begin, size, shape, dtype) {
    const isContinous = slice_util.isSliceContinous(shape, begin, size);
    const length = util.sizeFromShape(size);
    const xStrides = util.computeStrides(shape);
    if (isContinous) {
        const flatOffset = slice_util.computeFlatOffset(begin, xStrides);
        if (dtype === 'string') {
            return vals.slice(flatOffset, flatOffset + length);
        }
        return vals.subarray(flatOffset, flatOffset + length);
    }
    const decodedData = dtype === 'string' ?
        backend_util.fromUint8ToStringArray(vals) :
        vals;
    const inBuf = buffer(shape, dtype, decodedData);
    const outBuf = buffer(size, dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const outLoc = outBuf.indexToLoc(i);
        const inLoc = outLoc.map((idx, j) => idx + begin[j]);
        outBuf.set(inBuf.get(...inLoc), ...outLoc);
    }
    if (dtype === 'string') {
        return backend_util.fromStringArrayToUint8(outBuf.values);
    }
    return outBuf.values;
}
export function slice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, size } = attrs;
    assertNotComplex(x, 'slice');
    const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
    slice_util.assertParamsValid(x, $begin, $size);
    const vals = backend.data.get(x.dataId).values;
    const outVals = sliceImpl(vals, $begin, $size, x.shape, x.dtype);
    return backend.makeTensorInfo($size, x.dtype, outVals);
}
export const sliceConfig = {
    kernelName: Slice,
    backendName: 'cpu',
    kernelFunc: slice
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU2xpY2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtY3B1L3NyYy9rZXJuZWxzL1NsaWNlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQWlCLE1BQU0sRUFBc0MsS0FBSyxFQUFFLFVBQVUsRUFBbUQsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHeEwsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRTdDLE1BQU0sVUFBVSxTQUFTLENBQ3JCLElBQW1CLEVBQUUsS0FBZSxFQUFFLElBQWMsRUFBRSxLQUFlLEVBQ3JFLEtBQWU7SUFDakIsTUFBTSxXQUFXLEdBQUcsVUFBVSxDQUFDLGdCQUFnQixDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDcEUsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN4QyxNQUFNLFFBQVEsR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRTVDLElBQUksV0FBVyxFQUFFO1FBQ2YsTUFBTSxVQUFVLEdBQUcsVUFBVSxDQUFDLGlCQUFpQixDQUFDLEtBQUssRUFBRSxRQUFRLENBQUMsQ0FBQztRQUVqRSxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7WUFDdEIsT0FBUSxJQUFxQixDQUFDLEtBQUssQ0FBQyxVQUFVLEVBQUUsVUFBVSxHQUFHLE1BQU0sQ0FBQyxDQUFDO1NBQ3RFO1FBRUQsT0FBUSxJQUFtQixDQUFDLFFBQVEsQ0FBQyxVQUFVLEVBQUUsVUFBVSxHQUFHLE1BQU0sQ0FBQyxDQUFDO0tBQ3ZFO0lBRUQsTUFBTSxXQUFXLEdBQUcsS0FBSyxLQUFLLFFBQVEsQ0FBQyxDQUFDO1FBQ3BDLFlBQVksQ0FBQyxzQkFBc0IsQ0FBQyxJQUFvQixDQUFDLENBQUMsQ0FBQztRQUMzRCxJQUFrQixDQUFDO0lBRXZCLE1BQU0sS0FBSyxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQ2hELE1BQU0sTUFBTSxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDbkMsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDcEMsTUFBTSxNQUFNLEdBQUcsTUFBTSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNwQyxNQUFNLEtBQUssR0FBRyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBVyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzdELE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxHQUFHLEtBQUssQ0FBQyxFQUFFLEdBQUcsTUFBTSxDQUFDLENBQUM7S0FDNUM7SUFFRCxJQUFJLEtBQUssS0FBSyxRQUFRLEVBQUU7UUFDdEIsT0FBTyxZQUFZLENBQUMsc0JBQXNCLENBQUMsTUFBTSxDQUFDLE1BQWtCLENBQUMsQ0FBQztLQUN2RTtJQUNELE9BQU8sTUFBTSxDQUFDLE1BQW9CLENBQUM7QUFDckMsQ0FBQztBQUVELE1BQU0sVUFBVSxLQUFLLENBQ2pCLElBQXVFO0lBRXpFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ25CLE1BQU0sRUFBQyxLQUFLLEVBQUUsSUFBSSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRTVCLGdCQUFnQixDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUU3QixNQUFNLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxHQUFHLFVBQVUsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3BFLFVBQVUsQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLEVBQUUsTUFBTSxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBRS9DLE1BQU0sSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQyxNQUFNLENBQUM7SUFDL0MsTUFBTSxPQUFPLEdBQUcsU0FBUyxDQUFDLElBQUksRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ2pFLE9BQU8sT0FBTyxDQUFDLGNBQWMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxPQUFPLENBQUMsQ0FBQztBQUN6RCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFpQjtJQUN2QyxVQUFVLEVBQUUsS0FBSztJQUNqQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsS0FBOEI7Q0FDM0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIEJhY2tlbmRWYWx1ZXMsIGJ1ZmZlciwgRGF0YVR5cGUsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgU2xpY2UsIHNsaWNlX3V0aWwsIFNsaWNlQXR0cnMsIFNsaWNlSW5wdXRzLCBUZW5zb3JJbmZvLCBUeXBlZEFycmF5LCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge01hdGhCYWNrZW5kQ1BVfSBmcm9tICcuLi9iYWNrZW5kX2NwdSc7XG5pbXBvcnQge2Fzc2VydE5vdENvbXBsZXh9IGZyb20gJy4uL2NwdV91dGlsJztcblxuZXhwb3J0IGZ1bmN0aW9uIHNsaWNlSW1wbChcbiAgICB2YWxzOiBCYWNrZW5kVmFsdWVzLCBiZWdpbjogbnVtYmVyW10sIHNpemU6IG51bWJlcltdLCBzaGFwZTogbnVtYmVyW10sXG4gICAgZHR5cGU6IERhdGFUeXBlKTogQmFja2VuZFZhbHVlcyB7XG4gIGNvbnN0IGlzQ29udGlub3VzID0gc2xpY2VfdXRpbC5pc1NsaWNlQ29udGlub3VzKHNoYXBlLCBiZWdpbiwgc2l6ZSk7XG4gIGNvbnN0IGxlbmd0aCA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaXplKTtcbiAgY29uc3QgeFN0cmlkZXMgPSB1dGlsLmNvbXB1dGVTdHJpZGVzKHNoYXBlKTtcblxuICBpZiAoaXNDb250aW5vdXMpIHtcbiAgICBjb25zdCBmbGF0T2Zmc2V0ID0gc2xpY2VfdXRpbC5jb21wdXRlRmxhdE9mZnNldChiZWdpbiwgeFN0cmlkZXMpO1xuXG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgcmV0dXJuICh2YWxzIGFzIFVpbnQ4QXJyYXlbXSkuc2xpY2UoZmxhdE9mZnNldCwgZmxhdE9mZnNldCArIGxlbmd0aCk7XG4gICAgfVxuXG4gICAgcmV0dXJuICh2YWxzIGFzIFR5cGVkQXJyYXkpLnN1YmFycmF5KGZsYXRPZmZzZXQsIGZsYXRPZmZzZXQgKyBsZW5ndGgpO1xuICB9XG5cbiAgY29uc3QgZGVjb2RlZERhdGEgPSBkdHlwZSA9PT0gJ3N0cmluZycgP1xuICAgICAgYmFja2VuZF91dGlsLmZyb21VaW50OFRvU3RyaW5nQXJyYXkodmFscyBhcyBVaW50OEFycmF5W10pIDpcbiAgICAgIHZhbHMgYXMgVHlwZWRBcnJheTtcblxuICBjb25zdCBpbkJ1ZiA9IGJ1ZmZlcihzaGFwZSwgZHR5cGUsIGRlY29kZWREYXRhKTtcbiAgY29uc3Qgb3V0QnVmID0gYnVmZmVyKHNpemUsIGR0eXBlKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBvdXRCdWYuc2l6ZTsgKytpKSB7XG4gICAgY29uc3Qgb3V0TG9jID0gb3V0QnVmLmluZGV4VG9Mb2MoaSk7XG4gICAgY29uc3QgaW5Mb2MgPSBvdXRMb2MubWFwKChpZHg6IG51bWJlciwgaikgPT4gaWR4ICsgYmVnaW5bal0pO1xuICAgIG91dEJ1Zi5zZXQoaW5CdWYuZ2V0KC4uLmluTG9jKSwgLi4ub3V0TG9jKTtcbiAgfVxuXG4gIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICByZXR1cm4gYmFja2VuZF91dGlsLmZyb21TdHJpbmdBcnJheVRvVWludDgob3V0QnVmLnZhbHVlcyBhcyBzdHJpbmdbXSk7XG4gIH1cbiAgcmV0dXJuIG91dEJ1Zi52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNsaWNlKFxuICAgIGFyZ3M6IHtpbnB1dHM6IFNsaWNlSW5wdXRzLCBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSwgYXR0cnM6IFNsaWNlQXR0cnN9KTpcbiAgICBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICBjb25zdCB7YmVnaW4sIHNpemV9ID0gYXR0cnM7XG5cbiAgYXNzZXJ0Tm90Q29tcGxleCh4LCAnc2xpY2UnKTtcblxuICBjb25zdCBbJGJlZ2luLCAkc2l6ZV0gPSBzbGljZV91dGlsLnBhcnNlU2xpY2VQYXJhbXMoeCwgYmVnaW4sIHNpemUpO1xuICBzbGljZV91dGlsLmFzc2VydFBhcmFtc1ZhbGlkKHgsICRiZWdpbiwgJHNpemUpO1xuXG4gIGNvbnN0IHZhbHMgPSBiYWNrZW5kLmRhdGEuZ2V0KHguZGF0YUlkKS52YWx1ZXM7XG4gIGNvbnN0IG91dFZhbHMgPSBzbGljZUltcGwodmFscywgJGJlZ2luLCAkc2l6ZSwgeC5zaGFwZSwgeC5kdHlwZSk7XG4gIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKCRzaXplLCB4LmR0eXBlLCBvdXRWYWxzKTtcbn1cblxuZXhwb3J0IGNvbnN0IHNsaWNlQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFNsaWNlLFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IHNsaWNlIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==