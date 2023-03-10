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
import { LRNGrad, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
export function lRNGrad(args) {
    const { inputs, backend, attrs } = args;
    const { x, y, dy } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    assertNotComplex(dy, 'LRNGrad');
    const dySize = util.sizeFromShape(dy.shape);
    const channels = dy.shape[3];
    const dyValues = backend.data.get(dy.dataId).values;
    const xValues = backend.data.get(x.dataId).values;
    const yValues = backend.data.get(y.dataId).values;
    const result = new Float32Array(dySize);
    const size = dySize;
    for (let offset = 0; offset < size; offset++) {
        const currentChannel = offset % channels;
        const depthBegin = (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
        const depthEnd = (offset - currentChannel) +
            Math.min(channels, currentChannel + depthRadius + 1);
        let norm = 0;
        for (let k = depthBegin; k < depthEnd; k++) {
            norm += Math.pow(xValues[k], 2);
        }
        norm = alpha * norm + bias;
        for (let k = depthBegin; k < depthEnd; k++) {
            let dyi = -2 * alpha * beta * xValues[k] * yValues[offset] / norm;
            if (offset === k) {
                dyi += Math.pow(norm, -beta);
            }
            dyi *= dyValues[offset];
            result[k] += dyi;
        }
    }
    return backend.makeTensorInfo(dy.shape, x.dtype, result);
}
// tslint:disable-next-line: variable-name
export const LRNGradConfig = {
    kernelName: LRNGrad,
    backendName: 'cpu',
    kernelFunc: lRNGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTFJOR3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvTFJOR3JhZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLE9BQU8sRUFBdUQsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHbkksT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRTdDLE1BQU0sVUFBVSxPQUFPLENBQ25CLElBQ3lFO0lBRTNFLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDMUIsTUFBTSxFQUFDLFdBQVcsRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBQyxHQUFHLEtBQUssQ0FBQztJQUUvQyxnQkFBZ0IsQ0FBQyxFQUFFLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFaEMsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFNUMsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM3QixNQUFNLFFBQVEsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUNsRSxNQUFNLE9BQU8sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUNoRSxNQUFNLE9BQU8sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUNoRSxNQUFNLE1BQU0sR0FBRyxJQUFJLFlBQVksQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUN4QyxNQUFNLElBQUksR0FBRyxNQUFNLENBQUM7SUFFcEIsS0FBSyxJQUFJLE1BQU0sR0FBRyxDQUFDLEVBQUUsTUFBTSxHQUFHLElBQUksRUFBRSxNQUFNLEVBQUUsRUFBRTtRQUM1QyxNQUFNLGNBQWMsR0FBRyxNQUFNLEdBQUcsUUFBUSxDQUFDO1FBQ3pDLE1BQU0sVUFBVSxHQUNaLENBQUMsTUFBTSxHQUFHLGNBQWMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLGNBQWMsR0FBRyxXQUFXLENBQUMsQ0FBQztRQUMxRSxNQUFNLFFBQVEsR0FBRyxDQUFDLE1BQU0sR0FBRyxjQUFjLENBQUM7WUFDdEMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLEVBQUUsY0FBYyxHQUFHLFdBQVcsR0FBRyxDQUFDLENBQUMsQ0FBQztRQUV6RCxJQUFJLElBQUksR0FBRyxDQUFDLENBQUM7UUFDYixLQUFLLElBQUksQ0FBQyxHQUFHLFVBQVUsRUFBRSxDQUFDLEdBQUcsUUFBUSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzFDLElBQUksSUFBSSxJQUFJLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztTQUNqQztRQUNELElBQUksR0FBRyxLQUFLLEdBQUcsSUFBSSxHQUFHLElBQUksQ0FBQztRQUUzQixLQUFLLElBQUksQ0FBQyxHQUFHLFVBQVUsRUFBRSxDQUFDLEdBQUcsUUFBUSxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzFDLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQyxHQUFHLEtBQUssR0FBRyxJQUFJLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsR0FBRyxJQUFJLENBQUM7WUFDbEUsSUFBSSxNQUFNLEtBQUssQ0FBQyxFQUFFO2dCQUNoQixHQUFHLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQzthQUM5QjtZQUNELEdBQUcsSUFBSSxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDeEIsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLEdBQUcsQ0FBQztTQUNsQjtLQUNGO0lBRUQsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLEVBQUUsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQztBQUMzRCxDQUFDO0FBRUQsMENBQTBDO0FBQzFDLE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBaUI7SUFDekMsVUFBVSxFQUFFLE9BQU87SUFDbkIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLE9BQWdDO0NBQzdDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBMUk5HcmFkLCBMUk5HcmFkQXR0cnMsIExSTkdyYWRJbnB1dHMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7YXNzZXJ0Tm90Q29tcGxleH0gZnJvbSAnLi4vY3B1X3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gbFJOR3JhZChcbiAgICBhcmdzOlxuICAgICAgICB7aW5wdXRzOiBMUk5HcmFkSW5wdXRzLCBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSwgYXR0cnM6IExSTkdyYWRBdHRyc30pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eCwgeSwgZHl9ID0gaW5wdXRzO1xuICBjb25zdCB7ZGVwdGhSYWRpdXMsIGJpYXMsIGFscGhhLCBiZXRhfSA9IGF0dHJzO1xuXG4gIGFzc2VydE5vdENvbXBsZXgoZHksICdMUk5HcmFkJyk7XG5cbiAgY29uc3QgZHlTaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKGR5LnNoYXBlKTtcblxuICBjb25zdCBjaGFubmVscyA9IGR5LnNoYXBlWzNdO1xuICBjb25zdCBkeVZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoZHkuZGF0YUlkKS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbiAgY29uc3QgeFZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCB5VmFsdWVzID0gYmFja2VuZC5kYXRhLmdldCh5LmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gIGNvbnN0IHJlc3VsdCA9IG5ldyBGbG9hdDMyQXJyYXkoZHlTaXplKTtcbiAgY29uc3Qgc2l6ZSA9IGR5U2l6ZTtcblxuICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCBzaXplOyBvZmZzZXQrKykge1xuICAgIGNvbnN0IGN1cnJlbnRDaGFubmVsID0gb2Zmc2V0ICUgY2hhbm5lbHM7XG4gICAgY29uc3QgZGVwdGhCZWdpbiA9XG4gICAgICAgIChvZmZzZXQgLSBjdXJyZW50Q2hhbm5lbCkgKyBNYXRoLm1heCgwLCBjdXJyZW50Q2hhbm5lbCAtIGRlcHRoUmFkaXVzKTtcbiAgICBjb25zdCBkZXB0aEVuZCA9IChvZmZzZXQgLSBjdXJyZW50Q2hhbm5lbCkgK1xuICAgICAgICBNYXRoLm1pbihjaGFubmVscywgY3VycmVudENoYW5uZWwgKyBkZXB0aFJhZGl1cyArIDEpO1xuXG4gICAgbGV0IG5vcm0gPSAwO1xuICAgIGZvciAobGV0IGsgPSBkZXB0aEJlZ2luOyBrIDwgZGVwdGhFbmQ7IGsrKykge1xuICAgICAgbm9ybSArPSBNYXRoLnBvdyh4VmFsdWVzW2tdLCAyKTtcbiAgICB9XG4gICAgbm9ybSA9IGFscGhhICogbm9ybSArIGJpYXM7XG5cbiAgICBmb3IgKGxldCBrID0gZGVwdGhCZWdpbjsgayA8IGRlcHRoRW5kOyBrKyspIHtcbiAgICAgIGxldCBkeWkgPSAtMiAqIGFscGhhICogYmV0YSAqIHhWYWx1ZXNba10gKiB5VmFsdWVzW29mZnNldF0gLyBub3JtO1xuICAgICAgaWYgKG9mZnNldCA9PT0gaykge1xuICAgICAgICBkeWkgKz0gTWF0aC5wb3cobm9ybSwgLWJldGEpO1xuICAgICAgfVxuICAgICAgZHlpICo9IGR5VmFsdWVzW29mZnNldF07XG4gICAgICByZXN1bHRba10gKz0gZHlpO1xuICAgIH1cbiAgfVxuXG4gIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKGR5LnNoYXBlLCB4LmR0eXBlLCByZXN1bHQpO1xufVxuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IHZhcmlhYmxlLW5hbWVcbmV4cG9ydCBjb25zdCBMUk5HcmFkQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IExSTkdyYWQsXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogbFJOR3JhZCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=