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
import { FusedBatchNorm, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
export function batchNorm(args) {
    const { inputs, backend, attrs } = args;
    const { x, scale, offset, mean, variance } = inputs;
    util.assert(mean.shape.length === variance.shape.length, () => 'Batch normalization gradient requires mean and variance to have ' +
        'equal ranks.');
    util.assert(offset == null || mean.shape.length === offset.shape.length, () => 'Batch normalization gradient requires mean and offset to have ' +
        'equal ranks.');
    util.assert(scale == null || mean.shape.length === scale.shape.length, () => 'Batch normalization gradient requires mean and scale to have ' +
        'equal ranks.');
    assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');
    let { varianceEpsilon } = attrs;
    if (varianceEpsilon == null) {
        varianceEpsilon = 0.001;
    }
    const xVals = backend.data.get(x.dataId).values;
    const mVals = backend.data.get(mean.dataId).values;
    const varVals = backend.data.get(variance.dataId).values;
    const sVals = scale ? backend.data.get(scale.dataId).values :
        new Float32Array([1]);
    const offVals = offset ?
        backend.data.get(offset.dataId).values :
        new Float32Array([0]);
    const outVals = new Float32Array(xVals.length);
    const offValsLength = offVals.length;
    const sValsLength = sVals.length;
    const varValsLength = varVals.length;
    const mValsLength = mVals.length;
    let offi = 0;
    let mi = 0;
    let si = 0;
    let vi = 0;
    for (let i = 0; i < xVals.length; ++i) {
        outVals[i] = offVals[offi++] +
            (xVals[i] - mVals[mi++]) * sVals[si++] /
                Math.sqrt(varVals[vi++] + varianceEpsilon);
        if (offi >= offValsLength) {
            offi = 0;
        }
        if (mi >= mValsLength) {
            mi = 0;
        }
        if (si >= sValsLength) {
            si = 0;
        }
        if (vi >= varValsLength) {
            vi = 0;
        }
    }
    return backend.makeTensorInfo(x.shape, x.dtype, outVals);
}
export const batchNormConfig = {
    kernelName: FusedBatchNorm,
    backendName: 'cpu',
    kernelFunc: batchNorm,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQmF0Y2hOb3JtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9CYXRjaE5vcm0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLGNBQWMsRUFBK0YsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHeEosT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRTdDLE1BQU0sVUFBVSxTQUFTLENBQUMsSUFJekI7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFFbEQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxRQUFRLENBQUMsS0FBSyxDQUFDLE1BQU0sRUFDM0MsR0FBRyxFQUFFLENBQUMsa0VBQWtFO1FBQ3BFLGNBQWMsQ0FBQyxDQUFDO0lBQ3hCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxNQUFNLENBQUMsS0FBSyxDQUFDLE1BQU0sRUFDM0QsR0FBRyxFQUFFLENBQUMsZ0VBQWdFO1FBQ2xFLGNBQWMsQ0FBQyxDQUFDO0lBQ3hCLElBQUksQ0FBQyxNQUFNLENBQ1AsS0FBSyxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sRUFDekQsR0FBRyxFQUFFLENBQUMsK0RBQStEO1FBQ2pFLGNBQWMsQ0FBQyxDQUFDO0lBRXhCLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxF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QUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsU0FBa0M7Q0FDL0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtGdXNlZEJhdGNoTm9ybSwgRnVzZWRCYXRjaE5vcm1BdHRycywgRnVzZWRCYXRjaE5vcm1JbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgVHlwZWRBcnJheSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtNYXRoQmFja2VuZENQVX0gZnJvbSAnLi4vYmFja2VuZF9jcHUnO1xuaW1wb3J0IHthc3NlcnROb3RDb21wbGV4fSBmcm9tICcuLi9jcHVfdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBiYXRjaE5vcm0oYXJnczoge1xuICBpbnB1dHM6IEZ1c2VkQmF0Y2hOb3JtSW5wdXRzLFxuICBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSxcbiAgYXR0cnM6IEZ1c2VkQmF0Y2hOb3JtQXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3gsIHNjYWxlLCBvZmZzZXQsIG1lYW4sIHZhcmlhbmNlfSA9IGlucHV0cztcblxuICB1dGlsLmFzc2VydChcbiAgICAgIG1lYW4uc2hhcGUubGVuZ3RoID09PSB2YXJpYW5jZS5zaGFwZS5sZW5ndGgsXG4gICAgICAoKSA9PiAnQmF0Y2ggbm9ybWFsaXphdGlvbiBncmFkaWVudCByZXF1aXJlcyBtZWFuIGFuZCB2YXJpYW5jZSB0byBoYXZlICcgK1xuICAgICAgICAgICdlcXVhbCByYW5rcy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBvZmZzZXQgPT0gbnVsbCB8fCBtZWFuLnNoYXBlLmxlbmd0aCA9PT0gb2Zmc2V0LnNoYXBlLmxlbmd0aCxcbiAgICAgICgpID0+ICdCYXRjaCBub3JtYWxpemF0aW9uIGdyYWRpZW50IHJlcXVpcmVzIG1lYW4gYW5kIG9mZnNldCB0byBoYXZlICcgK1xuICAgICAgICAgICdlcXVhbCByYW5rcy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBzY2FsZSA9PSBudWxsIHx8IG1lYW4uc2hhcGUubGVuZ3RoID09PSBzY2FsZS5zaGFwZS5sZW5ndGgsXG4gICAgICAoKSA9PiAnQmF0Y2ggbm9ybWFsaXphdGlvbiBncmFkaWVudCByZXF1aXJlcyBtZWFuIGFuZCBzY2FsZSB0byBoYXZlICcgK1xuICAgICAgICAgICdlcXVhbCByYW5rcy4nKTtcblxuICBhc3NlcnROb3RDb21wbGV4KFt4LCBtZWFuLCB2YXJpYW5jZSwgc2NhbGUsIG9mZnNldF0sICdiYXRjaE5vcm0nKTtcblxuICBsZXQge3ZhcmlhbmNlRXBzaWxvbn0gPSBhdHRycztcbiAgaWYgKHZhcmlhbmNlRXBzaWxvbiA9PSBudWxsKSB7XG4gICAgdmFyaWFuY2VFcHNpbG9uID0gMC4wMDE7XG4gIH1cblxuICBjb25zdCB4VmFscyA9IGJhY2tlbmQuZGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCBtVmFscyA9IGJhY2tlbmQuZGF0YS5nZXQobWVhbi5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCB2YXJWYWxzID0gYmFja2VuZC5kYXRhLmdldCh2YXJpYW5jZS5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCBzVmFscyA9IHNjYWxlID8gYmFja2VuZC5kYXRhLmdldChzY2FsZS5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5IDpcbiAgICAgICAgICAgICAgICAgICAgICAgIG5ldyBGbG9hdDMyQXJyYXkoWzFdKTtcbiAgY29uc3Qgb2ZmVmFscyA9IG9mZnNldCA/XG4gICAgICBiYWNrZW5kLmRhdGEuZ2V0KG9mZnNldC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5IDpcbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkoWzBdKTtcbiAgY29uc3Qgb3V0VmFscyA9IG5ldyBGbG9hdDMyQXJyYXkoeFZhbHMubGVuZ3RoKTtcblxuICBjb25zdCBvZmZWYWxzTGVuZ3RoID0gb2ZmVmFscy5sZW5ndGg7XG4gIGNvbnN0IHNWYWxzTGVuZ3RoID0gc1ZhbHMubGVuZ3RoO1xuICBjb25zdCB2YXJWYWxzTGVuZ3RoID0gdmFyVmFscy5sZW5ndGg7XG4gIGNvbnN0IG1WYWxzTGVuZ3RoID0gbVZhbHMubGVuZ3RoO1xuXG4gIGxldCBvZmZpID0gMDtcbiAgbGV0IG1pID0gMDtcbiAgbGV0IHNpID0gMDtcbiAgbGV0IHZpID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCB4VmFscy5sZW5ndGg7ICsraSkge1xuICAgIG91dFZhbHNbaV0gPSBvZmZWYWxzW29mZmkrK10gK1xuICAgICAgICAoeFZhbHNbaV0gLSBtVmFsc1ttaSsrXSkgKiBzVmFsc1tzaSsrXSAvXG4gICAgICAgICAgICBNYXRoLnNxcnQodmFyVmFsc1t2aSsrXSArIHZhcmlhbmNlRXBzaWxvbik7XG4gICAgaWYgKG9mZmkgPj0gb2ZmVmFsc0xlbmd0aCkge1xuICAgICAgb2ZmaSA9IDA7XG4gICAgfVxuICAgIGlmIChtaSA+PSBtVmFsc0xlbmd0aCkge1xuICAgICAgbWkgPSAwO1xuICAgIH1cbiAgICBpZiAoc2kgPj0gc1ZhbHNMZW5ndGgpIHtcbiAgICAgIHNpID0gMDtcbiAgICB9XG4gICAgaWYgKHZpID49IHZhclZhbHNMZW5ndGgpIHtcbiAgICAgIHZpID0gMDtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8oeC5zaGFwZSwgeC5kdHlwZSwgb3V0VmFscyk7XG59XG5cbmV4cG9ydCBjb25zdCBiYXRjaE5vcm1Db25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogRnVzZWRCYXRjaE5vcm0sXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogYmF0Y2hOb3JtIGFzIHVua25vd24gYXMgS2VybmVsRnVuYyxcbn07XG4iXX0=