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
import { kernel_impls, NonMaxSuppressionV3 } from '@tensorflow/tfjs-core';
const nonMaxSuppressionV3Impl = kernel_impls.nonMaxSuppressionV3Impl;
import { assertNotComplex } from '../cpu_util';
export function nonMaxSuppressionV3(args) {
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold } = attrs;
    assertNotComplex(boxes, 'NonMaxSuppression');
    const boxesVals = backend.data.get(boxes.dataId).values;
    const scoresVals = backend.data.get(scores.dataId).values;
    const { selectedIndices } = nonMaxSuppressionV3Impl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    return backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices));
}
export const nonMaxSuppressionV3Config = {
    kernelName: NonMaxSuppressionV3,
    backendName: 'cpu',
    kernelFunc: nonMaxSuppressionV3
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTm9uTWF4U3VwcHJlc3Npb25WMy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvTm9uTWF4U3VwcHJlc3Npb25WMy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUE0QixtQkFBbUIsRUFBOEUsTUFBTSx1QkFBdUIsQ0FBQztBQUUvSyxNQUFNLHVCQUF1QixHQUFHLFlBQVksQ0FBQyx1QkFBdUIsQ0FBQztBQUdyRSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFN0MsTUFBTSxVQUFVLG1CQUFtQixDQUFDLElBSW5DO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxLQUFLLEVBQUUsTUFBTSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQy9CLE1BQU0sRUFBQyxhQUFhLEVBQUUsWUFBWSxFQUFFLGNBQWMsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUU1RCxnQkFBZ0IsQ0FBQyxLQUFLLEVBQUUsbUJBQW1CLENBQUMsQ0FBQztJQUU3QyxNQUFNLFNBQVMsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUN0RSxNQUFNLFVBQVUsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUV4RSxNQUFNLEVBQUMsZUFBZSxFQUFDLEdBQUcsdUJBQXVCLENBQzdDLFNBQVMsRUFBRSxVQUFVLEVBQUUsYUFBYSxFQUFFLFlBQVksRUFBRSxjQUFjLENBQUMsQ0FBQztJQUV4RSxPQUFPLE9BQU8sQ0FBQyxjQUFjLENBQ3pCLENBQUMsZUFBZSxDQUFDLE1BQU0sQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLFVBQVUsQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDO0FBQzFFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSx5QkFBeUIsR0FBaUI7SUFDckQsVUFBVSxFQUFFLG1CQUFtQjtJQUMvQixXQUFXLEVBQUUsS0FBSztJQUNsQixVQUFVLEVBQUUsbUJBQTRDO0NBQ3pELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7a2VybmVsX2ltcGxzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIE5vbk1heFN1cHByZXNzaW9uVjMsIE5vbk1heFN1cHByZXNzaW9uVjNBdHRycywgTm9uTWF4U3VwcHJlc3Npb25WM0lucHV0cywgVGVuc29ySW5mbywgVHlwZWRBcnJheX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuY29uc3Qgbm9uTWF4U3VwcHJlc3Npb25WM0ltcGwgPSBrZXJuZWxfaW1wbHMubm9uTWF4U3VwcHJlc3Npb25WM0ltcGw7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7YXNzZXJ0Tm90Q29tcGxleH0gZnJvbSAnLi4vY3B1X3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gbm9uTWF4U3VwcHJlc3Npb25WMyhhcmdzOiB7XG4gIGlucHV0czogTm9uTWF4U3VwcHJlc3Npb25WM0lucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRDUFUsXG4gIGF0dHJzOiBOb25NYXhTdXBwcmVzc2lvblYzQXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge2JveGVzLCBzY29yZXN9ID0gaW5wdXRzO1xuICBjb25zdCB7bWF4T3V0cHV0U2l6ZSwgaW91VGhyZXNob2xkLCBzY29yZVRocmVzaG9sZH0gPSBhdHRycztcblxuICBhc3NlcnROb3RDb21wbGV4KGJveGVzLCAnTm9uTWF4U3VwcHJlc3Npb24nKTtcblxuICBjb25zdCBib3hlc1ZhbHMgPSBiYWNrZW5kLmRhdGEuZ2V0KGJveGVzLmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gIGNvbnN0IHNjb3Jlc1ZhbHMgPSBiYWNrZW5kLmRhdGEuZ2V0KHNjb3Jlcy5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuXG4gIGNvbnN0IHtzZWxlY3RlZEluZGljZXN9ID0gbm9uTWF4U3VwcHJlc3Npb25WM0ltcGwoXG4gICAgICBib3hlc1ZhbHMsIHNjb3Jlc1ZhbHMsIG1heE91dHB1dFNpemUsIGlvdVRocmVzaG9sZCwgc2NvcmVUaHJlc2hvbGQpO1xuXG4gIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKFxuICAgICAgW3NlbGVjdGVkSW5kaWNlcy5sZW5ndGhdLCAnaW50MzInLCBuZXcgSW50MzJBcnJheShzZWxlY3RlZEluZGljZXMpKTtcbn1cblxuZXhwb3J0IGNvbnN0IG5vbk1heFN1cHByZXNzaW9uVjNDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTm9uTWF4U3VwcHJlc3Npb25WMyxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiBub25NYXhTdXBwcmVzc2lvblYzIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==