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
import { backend_util, DepthwiseConv2dNativeBackpropInput, TensorBuffer, util } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
export function depthwiseConv2dNativeBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { strides, dilations, pad, dimRoundingMode, inputShape } = attrs;
    assertNotComplex([dy, filter], 'depthwiseConv2DNativeBackpropInput');
    const dyStrides = util.computeStrides(dy.shape);
    const filterStrides = util.computeStrides(filter.shape);
    const convInfo = backend_util.computeConv2DInfo(inputShape, filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    const dx = new TensorBuffer(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2] = dx.strides;
    const dyValues = backend.data.get(dy.dataId).values;
    const [dyS0, dyS1, dyS2] = dyStrides;
    const fltValues = backend.data.get(filter.dataId).values;
    const [fltS0, fltS1, fltS2] = filterStrides;
    const { batchSize, filterHeight, filterWidth, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, strideHeight, strideWidth } = convInfo;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const chMul = outChannels / inChannels;
    for (let b = 0; b < batchSize; ++b) {
        for (let d1 = 0; d1 < inChannels; ++d1) {
            for (let xR = 0; xR < inHeight; ++xR) {
                const xRCorner = xR - topPad;
                const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                for (let xC = 0; xC < inWidth; ++xC) {
                    const xCCorner = xC - leftPad;
                    const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                    const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                    let dotProd = 0;
                    for (let yR = xRMin; yR < yRMax; ++yR) {
                        const wR = yR * strideHeight - xRCorner;
                        for (let yC = xCMin; yC < yCMax; ++yC) {
                            const wC = yC * strideWidth - xCCorner;
                            const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                            const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                            for (let dm = 0; dm < chMul; ++dm) {
                                const d2 = d1 * chMul + dm;
                                const pixel = dyValues[dyOffset + d2];
                                const weight = fltValues[fltOffset + dm];
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
export const depthwiseConv2dNativeBackpropInputConfig = {
    kernelName: DepthwiseConv2dNativeBackpropInput,
    backendName: 'cpu',
    kernelFunc: depthwiseConv2dNativeBackpropInput
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvRGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFFLGtDQUFrQyxFQUErRyxZQUFZLEVBQTBCLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBR2hQLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUU3QyxNQUFNLFVBQVUsa0NBQWtDLENBQUMsSUFJbEQ7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLEVBQUUsRUFBRSxNQUFNLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDNUIsTUFBTSxFQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBRSxVQUFVLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFckUsZ0JBQWdCLENBQUMsQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLEVBQUUsb0NBQW9DLENBQUMsQ0FBQztJQUVyRSxNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUNoRCxNQUFNLGFBQWEsR0FBRyxJQUFJLENBQUMsY0FBYyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUV4RCxNQUFNLFFBQVEsR0FBRyxZQUFZLENBQUMsaUJBQWlCLENBQzNDLFVBQVUsRUFBRSxNQUFNLENBQUMsS0FBeUMsRUFBRSxPQUFPLEVBQ3JFLFNBQVMsRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFFLElBQUksQ0FBQyxlQUFlLENBQUMs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MkQ7Q0FDeEUsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIERlcHRod2lzZUNvbnYyZE5hdGl2ZUJhY2twcm9wSW5wdXQsIERlcHRod2lzZUNvbnYyZE5hdGl2ZUJhY2twcm9wSW5wdXRBdHRycywgRGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dElucHV0cywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBUZW5zb3JCdWZmZXIsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcbmltcG9ydCB7YXNzZXJ0Tm90Q29tcGxleH0gZnJvbSAnLi4vY3B1X3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gZGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dChhcmdzOiB7XG4gIGlucHV0czogRGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dElucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRDUFUsXG4gIGF0dHJzOiBEZXB0aHdpc2VDb252MmROYXRpdmVCYWNrcHJvcElucHV0QXR0cnNcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge2R5LCBmaWx0ZXJ9ID0gaW5wdXRzO1xuICBjb25zdCB7c3RyaWRlcywgZGlsYXRpb25zLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSwgaW5wdXRTaGFwZX0gPSBhdHRycztcblxuICBhc3NlcnROb3RDb21wbGV4KFtkeSwgZmlsdGVyXSwgJ2RlcHRod2lzZUNvbnYyRE5hdGl2ZUJhY2twcm9wSW5wdXQnKTtcblxuICBjb25zdCBkeVN0cmlkZXMgPSB1dGlsLmNvbXB1dGVTdHJpZGVzKGR5LnNoYXBlKTtcbiAgY29uc3QgZmlsdGVyU3RyaWRlcyA9IHV0aWwuY29tcHV0ZVN0cmlkZXMoZmlsdGVyLnNoYXBlKTtcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlQ29udjJESW5mbyhcbiAgICAgIGlucHV0U2hhcGUsIGZpbHRlci5zaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgc3RyaWRlcyxcbiAgICAgIGRpbGF0aW9ucywgcGFkLCBkaW1Sb3VuZGluZ01vZGUsIHRydWUgLyogZGVwdGh3aXNlICovKTtcblxuICBjb25zdCBkeCA9IG5ldyBUZW5zb3JCdWZmZXIoY29udkluZm8uaW5TaGFwZSwgJ2Zsb2F0MzInKTtcbiAgY29uc3QgZHhWYWx1ZXMgPSBkeC52YWx1ZXM7XG4gIGNvbnN0IFtkeFMwLCBkeFMxLCBkeFMyXSA9IGR4LnN0cmlkZXM7XG4gIGNvbnN0IGR5VmFsdWVzID0gYmFja2VuZC5kYXRhLmdldChkeS5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICBjb25zdCBbZHlTMCwgZHlTMSwgZHlTMl0gPSBkeVN0cmlkZXM7XG4gIGNvbnN0IGZsdFZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQoZmlsdGVyLmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gIGNvbnN0IFtmbHRTMCwgZmx0UzEsIGZsdFMyXSA9IGZpbHRlclN0cmlkZXM7XG4gIGNvbnN0IHtcbiAgICBiYXRjaFNpemUsXG4gICAgZmlsdGVySGVpZ2h0LFxuICAgIGZpbHRlcldpZHRoLFxuICAgIGluQ2hhbm5lbHMsXG4gICAgaW5IZWlnaHQsXG4gICAgaW5XaWR0aCxcbiAgICBvdXRDaGFubmVscyxcbiAgICBvdXRIZWlnaHQsXG4gICAgb3V0V2lkdGgsXG4gICAgc3RyaWRlSGVpZ2h0LFxuICAgIHN0cmlkZVdpZHRoXG4gIH0gPSBjb252SW5mbztcbiAgY29uc3QgdG9wUGFkID0gZmlsdGVySGVpZ2h0IC0gMSAtIGNvbnZJbmZvLnBhZEluZm8udG9wO1xuICBjb25zdCBsZWZ0UGFkID0gZmlsdGVyV2lkdGggLSAxIC0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuICBjb25zdCBjaE11bCA9IG91dENoYW5uZWxzIC8gaW5DaGFubmVscztcblxuICBmb3IgKGxldCBiID0gMDsgYiA8IGJhdGNoU2l6ZTsgKytiKSB7XG4gICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IGluQ2hhbm5lbHM7ICsrZDEpIHtcbiAgICAgIGZvciAobGV0IHhSID0gMDsgeFIgPCBpbkhlaWdodDsgKyt4Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHhSIC0gdG9wUGFkO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCh4UkNvcm5lciAvIHN0cmlkZUhlaWdodCkpO1xuICAgICAgICBjb25zdCB5Uk1heCA9XG4gICAgICAgICAgICBNYXRoLm1pbihvdXRIZWlnaHQsIChmaWx0ZXJIZWlnaHQgKyB4UkNvcm5lcikgLyBzdHJpZGVIZWlnaHQpO1xuXG4gICAgICAgIGZvciAobGV0IHhDID0gMDsgeEMgPCBpbldpZHRoOyArK3hDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB4QyAtIGxlZnRQYWQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoeENDb3JuZXIgLyBzdHJpZGVXaWR0aCkpO1xuICAgICAgICAgIGNvbnN0IHlDTWF4ID1cbiAgICAgICAgICAgICAgTWF0aC5taW4ob3V0V2lkdGgsIChmaWx0ZXJXaWR0aCArIHhDQ29ybmVyKSAvIHN0cmlkZVdpZHRoKTtcblxuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB5UiA9IHhSTWluOyB5UiA8IHlSTWF4OyArK3lSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHlSICogc3RyaWRlSGVpZ2h0IC0geFJDb3JuZXI7XG5cbiAgICAgICAgICAgIGZvciAobGV0IHlDID0geENNaW47IHlDIDwgeUNNYXg7ICsreUMpIHtcbiAgICAgICAgICAgICAgY29uc3Qgd0MgPSB5QyAqIHN0cmlkZVdpZHRoIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IGR5T2Zmc2V0ID0gZHlTMCAqIGIgKyBkeVMxICogeVIgKyBkeVMyICogeUM7XG4gICAgICAgICAgICAgIGNvbnN0IGZsdE9mZnNldCA9IGZsdFMwICogKGZpbHRlckhlaWdodCAtIDEgLSB3UikgK1xuICAgICAgICAgICAgICAgICAgZmx0UzEgKiAoZmlsdGVyV2lkdGggLSAxIC0gd0MpICsgZmx0UzIgKiBkMTtcblxuICAgICAgICAgICAgICBmb3IgKGxldCBkbSA9IDA7IGRtIDwgY2hNdWw7ICsrZG0pIHtcbiAgICAgICAgICAgICAgICBjb25zdCBkMiA9IGQxICogY2hNdWwgKyBkbTtcbiAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IGR5VmFsdWVzW2R5T2Zmc2V0ICsgZDJdO1xuICAgICAgICAgICAgICAgIGNvbnN0IHdlaWdodCA9IGZsdFZhbHVlc1tmbHRPZmZzZXQgKyBkbV07XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIHdlaWdodDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBkeFZhbHVlc1tkeFMwICogYiArIGR4UzEgKiB4UiArIGR4UzIgKiB4QyArIGQxXSA9IGRvdFByb2Q7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICByZXR1cm4gYmFja2VuZC5tYWtlVGVuc29ySW5mbyhkeC5zaGFwZSwgZHguZHR5cGUsIGR4LnZhbHVlcyk7XG59XG5cbmV4cG9ydCBjb25zdCBkZXB0aHdpc2VDb252MmROYXRpdmVCYWNrcHJvcElucHV0Q29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IERlcHRod2lzZUNvbnYyZE5hdGl2ZUJhY2twcm9wSW5wdXQsXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogZGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=