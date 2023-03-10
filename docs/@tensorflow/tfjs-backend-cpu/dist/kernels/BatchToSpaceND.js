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
import { backend_util, BatchToSpaceND } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { reshape } from './Reshape';
import { slice } from './Slice';
import { transpose } from './Transpose';
export function batchToSpaceND(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, crops } = attrs;
    assertNotComplex([x], 'batchToSpaceND');
    const prod = blockShape.reduce((a, b) => a * b);
    const reshaped = backend_util.getReshaped(x.shape, blockShape, prod);
    const permuted = backend_util.getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted = backend_util.getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords = backend_util.getSliceBeginCoords(crops, blockShape.length);
    const sliceSize = backend_util.getSliceSize(reshapedPermuted, crops, blockShape.length);
    const xReshaped = reshape({ inputs: { x }, backend, attrs: { shape: reshaped } });
    const xTransposed = transpose({ inputs: { x: xReshaped }, backend, attrs: { perm: permuted } });
    const xTransposedReshaped = reshape({ inputs: { x: xTransposed }, backend, attrs: { shape: reshapedPermuted } });
    const result = slice({
        inputs: { x: xTransposedReshaped },
        backend,
        attrs: { begin: sliceBeginCoords, size: sliceSize }
    });
    backend.disposeIntermediateTensorInfo(xReshaped);
    backend.disposeIntermediateTensorInfo(xTransposed);
    backend.disposeIntermediateTensorInfo(xTransposedReshaped);
    return result;
}
export const batchToSpaceNDConfig = {
    kernelName: BatchToSpaceND,
    backendName: 'cpu',
    kernelFunc: batchToSpaceND
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQmF0Y2hUb1NwYWNlTkQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtY3B1L3NyYy9rZXJuZWxzL0JhdGNoVG9TcGFjZU5ELnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQUUsY0FBYyxFQUFrRixNQUFNLHVCQUF1QixDQUFDO0FBR3BKLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUM3QyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUV0QyxNQUFNLFVBQVUsY0FBYyxDQUFDLElBSTlCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLFVBQVUsRUFBRSxLQUFLLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFbEMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBRXhDLE1BQU0sSUFBSSxHQUFHLFVBQVUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFFaEQsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLFVBQVUsRUFBRSxJQUFJLENBQUMsQ0FBQztJQUNyRSxNQUFNLFFBQVEsR0FBRyxZQUFZLENBQUMsV0FBVyxDQUFDLFFBQVEsQ0FBQyxNQUFNLEVBQUUsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQzlFLE1BQU0sZ0JBQWdCLEdBQ2xCLFlBQVksQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLFVBQVUsRUFBRSxJQUFJLENBQUMsQ0FBQztJQUNoRSxNQUFNLGdCQUFnQixHQUNsQixZQUFZLENBQUMsbUJBQW1CLENBQUMsS0FBSyxFQUFFLFVBQVUsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUMvRCxNQUFNLFNBQVMsR0FDWCxZQUFZLENBQUMsWUFBWSxDQUFDLGdCQUFnQixFQUFFLEtBQUssRUFBRSxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7SUFFMUUsTUFBTSxTQUFTLEdBQUcsT0FBTyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxRQUFRLEVBQUMsRUFBQyxDQUFDLENBQUM7SUFDNUUsTUFBTSxXQUFXLEdBQ2IsU0FBUyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLFNBQVMsRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxJQUFJLEVBQUUsUUFBUSxFQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQzFFLE1BQU0sbUJBQW1CLEdBQUcsT0FBTyxDQUMvQixFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxXQUFXLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLGdCQUFnQixFQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQzNFLE1BQU0sTUFBTSxHQUFHLEtBQUssQ0FBQztRQUNuQixNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsbUJBQW1CLEVBQUM7UUFDaEMsT0FBTztRQUNQLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxnQkFBZ0IsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFDO0tBQ2xELENBQUMsQ0FBQztJQUVILE9BQU8sQ0FBQyw2QkFBNkIsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUNqRCxPQUFPLENBQUMsNkJBQTZCLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDbkQsT0FBTyxDQUFDLDZCQUE2QixDQUFDLG1CQUFtQixDQUFDLENBQUM7SUFFM0QsT0FBTyxNQUFNLENBQUM7QUFDaEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLG9CQUFvQixHQUFpQjtJQUNoRCxVQUFVLEVBQUUsY0FBYztJQUMxQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsY0FBdUM7Q0FDcEQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIEJhdGNoVG9TcGFjZU5ELCBCYXRjaFRvU3BhY2VOREF0dHJzLCBCYXRjaFRvU3BhY2VORElucHV0cywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge01hdGhCYWNrZW5kQ1BVfSBmcm9tICcuLi9iYWNrZW5kX2NwdSc7XG5pbXBvcnQge2Fzc2VydE5vdENvbXBsZXh9IGZyb20gJy4uL2NwdV91dGlsJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9SZXNoYXBlJztcbmltcG9ydCB7c2xpY2V9IGZyb20gJy4vU2xpY2UnO1xuaW1wb3J0IHt0cmFuc3Bvc2V9IGZyb20gJy4vVHJhbnNwb3NlJztcblxuZXhwb3J0IGZ1bmN0aW9uIGJhdGNoVG9TcGFjZU5EKGFyZ3M6IHtcbiAgaW5wdXRzOiBCYXRjaFRvU3BhY2VORElucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRDUFUsXG4gIGF0dHJzOiBCYXRjaFRvU3BhY2VOREF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHt4fSA9IGlucHV0cztcbiAgY29uc3Qge2Jsb2NrU2hhcGUsIGNyb3BzfSA9IGF0dHJzO1xuXG4gIGFzc2VydE5vdENvbXBsZXgoW3hdLCAnYmF0Y2hUb1NwYWNlTkQnKTtcblxuICBjb25zdCBwcm9kID0gYmxvY2tTaGFwZS5yZWR1Y2UoKGEsIGIpID0+IGEgKiBiKTtcblxuICBjb25zdCByZXNoYXBlZCA9IGJhY2tlbmRfdXRpbC5nZXRSZXNoYXBlZCh4LnNoYXBlLCBibG9ja1NoYXBlLCBwcm9kKTtcbiAgY29uc3QgcGVybXV0ZWQgPSBiYWNrZW5kX3V0aWwuZ2V0UGVybXV0ZWQocmVzaGFwZWQubGVuZ3RoLCBibG9ja1NoYXBlLmxlbmd0aCk7XG4gIGNvbnN0IHJlc2hhcGVkUGVybXV0ZWQgPVxuICAgICAgYmFja2VuZF91dGlsLmdldFJlc2hhcGVkUGVybXV0ZWQoeC5zaGFwZSwgYmxvY2tTaGFwZSwgcHJvZCk7XG4gIGNvbnN0IHNsaWNlQmVnaW5Db29yZHMgPVxuICAgICAgYmFja2VuZF91dGlsLmdldFNsaWNlQmVnaW5Db29yZHMoY3JvcHMsIGJsb2NrU2hhcGUubGVuZ3RoKTtcbiAgY29uc3Qgc2xpY2VTaXplID1cbiAgICAgIGJhY2tlbmRfdXRpbC5nZXRTbGljZVNpemUocmVzaGFwZWRQZXJtdXRlZCwgY3JvcHMsIGJsb2NrU2hhcGUubGVuZ3RoKTtcblxuICBjb25zdCB4UmVzaGFwZWQgPSByZXNoYXBlKHtpbnB1dHM6IHt4fSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogcmVzaGFwZWR9fSk7XG4gIGNvbnN0IHhUcmFuc3Bvc2VkID1cbiAgICAgIHRyYW5zcG9zZSh7aW5wdXRzOiB7eDogeFJlc2hhcGVkfSwgYmFja2VuZCwgYXR0cnM6IHtwZXJtOiBwZXJtdXRlZH19KTtcbiAgY29uc3QgeFRyYW5zcG9zZWRSZXNoYXBlZCA9IHJlc2hhcGUoXG4gICAgICB7aW5wdXRzOiB7eDogeFRyYW5zcG9zZWR9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiByZXNoYXBlZFBlcm11dGVkfX0pO1xuICBjb25zdCByZXN1bHQgPSBzbGljZSh7XG4gICAgaW5wdXRzOiB7eDogeFRyYW5zcG9zZWRSZXNoYXBlZH0sXG4gICAgYmFja2VuZCxcbiAgICBhdHRyczoge2JlZ2luOiBzbGljZUJlZ2luQ29vcmRzLCBzaXplOiBzbGljZVNpemV9XG4gIH0pO1xuXG4gIGJhY2tlbmQuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oeFJlc2hhcGVkKTtcbiAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh4VHJhbnNwb3NlZCk7XG4gIGJhY2tlbmQuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8oeFRyYW5zcG9zZWRSZXNoYXBlZCk7XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGNvbnN0IGJhdGNoVG9TcGFjZU5EQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IEJhdGNoVG9TcGFjZU5ELFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IGJhdGNoVG9TcGFjZU5EIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==