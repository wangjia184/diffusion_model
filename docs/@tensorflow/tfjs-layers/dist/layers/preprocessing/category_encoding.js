/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { Layer } from '../../engine/topology';
import { serialization, tidy } from '@tensorflow/tfjs-core';
import { greater, greaterEqual, max, min } from '@tensorflow/tfjs-core';
import { getExactlyOneShape, getExactlyOneTensor } from '../../utils/types_utils';
import { ValueError } from '../../errors';
import * as K from '../../backend/tfjs_backend';
import * as utils from './preprocessing_utils';
export class CategoryEncoding extends Layer {
    constructor(args) {
        super(args);
        this.numTokens = args.numTokens;
        if (args.outputMode) {
            this.outputMode = args.outputMode;
        }
        else {
            this.outputMode = 'multiHot';
        }
    }
    getConfig() {
        const config = {
            'numTokens': this.numTokens,
            'outputMode': this.outputMode,
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (inputShape == null) {
            return [this.numTokens];
        }
        if (this.outputMode === 'oneHot' && inputShape[inputShape.length - 1] !== 1) {
            inputShape.push(this.numTokens);
            return inputShape;
        }
        inputShape[inputShape.length - 1] = this.numTokens;
        return inputShape;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = getExactlyOneTensor(inputs);
            if (inputs.dtype !== 'int32') {
                inputs = K.cast(inputs, 'int32');
            }
            let countWeights;
            if ((typeof kwargs['countWeights']) !== 'undefined') {
                if (this.outputMode !== 'count') {
                    throw new ValueError(`countWeights is not used when outputMode !== count.
              Received countWeights=${kwargs['countWeights']}`);
                }
                countWeights
                    = getExactlyOneTensor(kwargs['countWeights']);
            }
            const maxValue = max(inputs);
            const minValue = min(inputs);
            const greaterEqualMax = greater(this.numTokens, maxValue)
                .bufferSync().get(0);
            const greaterMin = greaterEqual(minValue, 0).bufferSync().get(0);
            if (!(greaterEqualMax && greaterMin)) {
                throw new ValueError('Input values must be between 0 < values <='
                    + ` numTokens with numTokens=${this.numTokens}`);
            }
            return utils.encodeCategoricalInputs(inputs, this.outputMode, this.numTokens, countWeights);
        });
    }
}
/** @nocollapse */
CategoryEncoding.className = 'CategoryEncoding';
serialization.registerClass(CategoryEncoding);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2F0ZWdvcnlfZW5jb2RpbmcuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL3ByZXByb2Nlc3NpbmcvY2F0ZWdvcnlfZW5jb2RpbmcudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCxPQUFPLEVBQWEsS0FBSyxFQUFFLE1BQU0sdUJBQXVCLENBQUM7QUFDekQsT0FBTyxFQUFFLGFBQWEsRUFBVSxJQUFJLEVBQXFCLE1BQU0sdUJBQXVCLENBQUM7QUFDdkYsT0FBTyxFQUFFLE9BQU8sRUFBRSxZQUFZLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXZFLE9BQU8sRUFBRSxrQkFBa0IsRUFBRSxtQkFBbUIsRUFBRSxNQUFNLHlCQUF5QixDQUFDO0FBRWxGLE9BQU8sRUFBRSxVQUFVLEVBQUUsTUFBTSxjQUFjLENBQUM7QUFDMUMsT0FBTyxLQUFLLENBQUMsTUFBTSw0QkFBNEIsQ0FBQztBQUNoRCxPQUFPLEtBQUssS0FBSyxNQUFNLHVCQUF1QixDQUFDO0FBUS9DLE1BQU0sT0FBTyxnQkFBaUIsU0FBUSxLQUFLO0lBTXpDLFlBQVksSUFBMEI7UUFDcEMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ1osSUFBSSxDQUFDLFNBQVMsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDO1FBRWhDLElBQUcsSUFBSSxDQUFDLFVBQVUsRUFBRTtZQUNwQixJQUFJLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUM7U0FDakM7YUFBTTtZQUNMLElBQUksQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDO1NBQzlCO0lBQ0gsQ0FBQztJQUVRLFNBQVM7UUFDaEIsTUFBTSxNQUFNLEdBQTZCO1lBQ3ZDLFdBQVcsRUFBRSxJQUFJLENBQUMsU0FBUztZQUMzQixZQUFZLEVBQUUsSUFBSSxDQUFDLFVBQVU7U0FDOUIsQ0FBQztRQUVGLE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRVEsa0JBQWtCLENBQUMsVUFBeUI7UUFDbkQsVUFBVSxHQUFHLGtCQUFrQixDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRTVDLElBQUcsVUFBVSxJQUFJLElBQUksRUFBRTtZQUNyQixPQUFPLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1NBQ3pCO1FBRUQsSUFBRyxJQUFJLENBQUMsVUFBVSxLQUFLLFFBQVEsSUFBSSxVQUFVLENBQUMsVUFBVSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUM7WUFDekUsVUFBVSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7WUFDaEMsT0FBTyxVQUFVLENBQUM7U0FDbkI7UUFFRCxVQUFVLENBQUMsVUFBVSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDO1FBQ25ELE9BQU8sVUFBVSxDQUFDO0lBQ3BCLENBQUM7SUFFUSxJQUFJLENBQUMsTUFBdUIsRUFBRSxNQUFjO1FBQ25ELE9BQU8sSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUViLE1BQU0sR0FBRyxtQkFBbUIsQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUNyQyxJQUFHLE1BQU0sQ0FBQyxLQUFLLEtBQUssT0FBTyxFQUFFO2dCQUMzQixNQUFNLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7YUFDcEM7WUFFQyxJQUFJLFlBQWlDLENBQUM7WUFFdEMsSUFBRyxDQUFDLE9BQU8sTUFBTSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEtBQUssV0FBVyxFQUFFO2dCQUVsRCxJQUFHLElBQUksQ0FBQyxVQUFVLEtBQUssT0FBTyxFQUFFO29CQUM5QixNQUFNLElBQUksVUFBVSxDQUNsQjtzQ0FDd0IsTUFBTSxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztpQkFDckQ7Z0JBRUQsWUFBWTtzQkFDUCxtQkFBbUIsQ0FBQyxNQUFNLENBQUMsY0FBYyxDQUFDLENBQXNCLENBQUM7YUFDdkU7WUFFRCxNQUFNLFFBQVEsR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDN0IsTUFBTSxRQUFRLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQzdCLE1BQU0sZUFBZSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsU0FBUyxFQUFFLFFBQVEsQ0FBQztpQkFDWixVQUFVLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFFakUsTUFBTSxVQUFVLEdBQUcsWUFBWSxDQUFDLFFBQVEsRUFBRSxDQUFDLENBQUMsQ0FBQyxVQUFVLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFFakUsSUFBRyxDQUFDLENBQUMsZUFBZSxJQUFJLFVBQVUsQ0FBQyxFQUFFO2dCQUVuQyxNQUFNLElBQUksVUFBVSxDQUFDLDRDQUE0QztzQkFDN0QsNkJBQTZCLElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FBQyxDQUFDO2FBQ3BEO1lBRUQsT0FBTyxLQUFLLENBQUMsdUJBQXVCLENBQUMsTUFBTSxFQUN6QyxJQUFJLENBQUMsVUFBVSxFQUFFLElBQUksQ0FBQyxTQUFTLEVBQUUsWUFBWSxDQUFDLENBQUM7UUFDckQsQ0FBQyxDQUFDLENBQUM7SUFDTCxDQUFDOztBQWpGRCxrQkFBa0I7QUFDWCwwQkFBUyxHQUFHLGtCQUFrQixDQUFDO0FBbUZ4QyxhQUFhLENBQUMsYUFBYSxDQUFDLGdCQUFnQixDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBDb2RlU21pdGggTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgeyBMYXllckFyZ3MsIExheWVyIH0gZnJvbSAnLi4vLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7IHNlcmlhbGl6YXRpb24sIFRlbnNvciwgdGlkeSwgVGVuc29yMUQsIFRlbnNvcjJEfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHsgZ3JlYXRlciwgZ3JlYXRlckVxdWFsLCBtYXgsIG1pbn0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7IFNoYXBlIH0gZnJvbSAnLi4vLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQgeyBnZXRFeGFjdGx5T25lU2hhcGUsIGdldEV4YWN0bHlPbmVUZW5zb3IgfSBmcm9tICcuLi8uLi91dGlscy90eXBlc191dGlscyc7XG5pbXBvcnQgeyBLd2FyZ3MgfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQgeyBWYWx1ZUVycm9yIH0gZnJvbSAnLi4vLi4vZXJyb3JzJztcbmltcG9ydCAqIGFzIEsgZnJvbSAnLi4vLi4vYmFja2VuZC90ZmpzX2JhY2tlbmQnO1xuaW1wb3J0ICogYXMgdXRpbHMgZnJvbSAnLi9wcmVwcm9jZXNzaW5nX3V0aWxzJztcbmltcG9ydCB7IE91dHB1dE1vZGUgfSBmcm9tICcuL3ByZXByb2Nlc3NpbmdfdXRpbHMnO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgQ2F0ZWdvcnlFbmNvZGluZ0FyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICBudW1Ub2tlbnM6IG51bWJlcjtcbiAgb3V0cHV0TW9kZT86IE91dHB1dE1vZGU7XG4gfVxuXG5leHBvcnQgY2xhc3MgQ2F0ZWdvcnlFbmNvZGluZyBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnQ2F0ZWdvcnlFbmNvZGluZyc7XG4gIHByaXZhdGUgcmVhZG9ubHkgbnVtVG9rZW5zOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgb3V0cHV0TW9kZTogT3V0cHV0TW9kZTtcblxuICBjb25zdHJ1Y3RvcihhcmdzOiBDYXRlZ29yeUVuY29kaW5nQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIHRoaXMubnVtVG9rZW5zID0gYXJncy5udW1Ub2tlbnM7XG5cbiAgICBpZihhcmdzLm91dHB1dE1vZGUpIHtcbiAgICB0aGlzLm91dHB1dE1vZGUgPSBhcmdzLm91dHB1dE1vZGU7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMub3V0cHV0TW9kZSA9ICdtdWx0aUhvdCc7XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICAnbnVtVG9rZW5zJzogdGhpcy5udW1Ub2tlbnMsXG4gICAgICAnb3V0cHV0TW9kZSc6IHRoaXMub3V0cHV0TW9kZSxcbiAgICB9O1xuXG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG5cbiAgb3ZlcnJpZGUgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICBpbnB1dFNoYXBlID0gZ2V0RXhhY3RseU9uZVNoYXBlKGlucHV0U2hhcGUpO1xuXG4gICAgaWYoaW5wdXRTaGFwZSA9PSBudWxsKSB7XG4gICAgICByZXR1cm4gW3RoaXMubnVtVG9rZW5zXTtcbiAgICB9XG5cbiAgICBpZih0aGlzLm91dHB1dE1vZGUgPT09ICdvbmVIb3QnICYmIGlucHV0U2hhcGVbaW5wdXRTaGFwZS5sZW5ndGggLSAxXSAhPT0gMSl7XG4gICAgICBpbnB1dFNoYXBlLnB1c2godGhpcy5udW1Ub2tlbnMpO1xuICAgICAgcmV0dXJuIGlucHV0U2hhcGU7XG4gICAgfVxuXG4gICAgaW5wdXRTaGFwZVtpbnB1dFNoYXBlLmxlbmd0aCAtIDFdID0gdGhpcy5udW1Ub2tlbnM7XG4gICAgcmV0dXJuIGlucHV0U2hhcGU7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcltdfFRlbnNvciB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuXG4gICAgICAgIGlucHV0cyA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICAgICAgaWYoaW5wdXRzLmR0eXBlICE9PSAnaW50MzInKSB7XG4gICAgICAgICAgaW5wdXRzID0gSy5jYXN0KGlucHV0cywgJ2ludDMyJyk7XG4gICAgICB9XG5cbiAgICAgICAgbGV0IGNvdW50V2VpZ2h0czogVGVuc29yMUQgfCBUZW5zb3IyRDtcblxuICAgICAgICBpZigodHlwZW9mIGt3YXJnc1snY291bnRXZWlnaHRzJ10pICE9PSAndW5kZWZpbmVkJykge1xuXG4gICAgICAgICAgaWYodGhpcy5vdXRwdXRNb2RlICE9PSAnY291bnQnKSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgVmFsdWVFcnJvcihcbiAgICAgICAgICAgICAgYGNvdW50V2VpZ2h0cyBpcyBub3QgdXNlZCB3aGVuIG91dHB1dE1vZGUgIT09IGNvdW50LlxuICAgICAgICAgICAgICBSZWNlaXZlZCBjb3VudFdlaWdodHM9JHtrd2FyZ3NbJ2NvdW50V2VpZ2h0cyddfWApO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGNvdW50V2VpZ2h0c1xuICAgICAgICAgICAgPSAgZ2V0RXhhY3RseU9uZVRlbnNvcihrd2FyZ3NbJ2NvdW50V2VpZ2h0cyddKSBhcyBUZW5zb3IxRHxUZW5zb3IyRDtcbiAgICAgICAgfVxuXG4gICAgICAgIGNvbnN0IG1heFZhbHVlID0gbWF4KGlucHV0cyk7XG4gICAgICAgIGNvbnN0IG1pblZhbHVlID0gbWluKGlucHV0cyk7XG4gICAgICAgIGNvbnN0IGdyZWF0ZXJFcXVhbE1heCA9IGdyZWF0ZXIodGhpcy5udW1Ub2tlbnMsIG1heFZhbHVlKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC5idWZmZXJTeW5jKCkuZ2V0KDApO1xuXG4gICAgICAgIGNvbnN0IGdyZWF0ZXJNaW4gPSBncmVhdGVyRXF1YWwobWluVmFsdWUsIDApLmJ1ZmZlclN5bmMoKS5nZXQoMCk7XG5cbiAgICAgICAgaWYoIShncmVhdGVyRXF1YWxNYXggJiYgZ3JlYXRlck1pbikpIHtcblxuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKCdJbnB1dCB2YWx1ZXMgbXVzdCBiZSBiZXR3ZWVuIDAgPCB2YWx1ZXMgPD0nXG4gICAgICAgICAgICArIGAgbnVtVG9rZW5zIHdpdGggbnVtVG9rZW5zPSR7dGhpcy5udW1Ub2tlbnN9YCk7XG4gICAgICAgIH1cblxuICAgICAgICByZXR1cm4gdXRpbHMuZW5jb2RlQ2F0ZWdvcmljYWxJbnB1dHMoaW5wdXRzLFxuICAgICAgICAgIHRoaXMub3V0cHV0TW9kZSwgdGhpcy5udW1Ub2tlbnMsIGNvdW50V2VpZ2h0cyk7XG4gICAgfSk7XG4gIH1cbn1cblxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKENhdGVnb3J5RW5jb2RpbmcpO1xuIl19