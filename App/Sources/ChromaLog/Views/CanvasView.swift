import SwiftUI
import UniformTypeIdentifiers

/// Main stage (spec §6): plate image + annotation canvas, floating glass
/// toolbar, and the reaction time-course filmstrip. M0 displays the image and
/// accepts drag-and-drop; annotation lands in M1.
struct CanvasView: View {
    @EnvironmentObject private var store: AppStore
    @State private var isTargeted = false

    var body: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
                .ignoresSafeArea()

            if let image = store.plateImage {
                Group {
                    if store.showDigitalPlate {
                        DigitalPlateView()
                    } else {
                        PlateCanvas(image: image)
                    }
                }
                .padding(40)
                .padding(.bottom, 120) // room for toolbar + filmstrip
            } else {
                EmptyCanvas(isTargeted: isTargeted)
            }

            VStack(spacing: 0) {
                Spacer()
                if store.isSpotMode {
                    LabelPicker()
                        .padding(.bottom, 10)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                FloatingToolbar()
            }
            .padding(.bottom, 12)
            .animation(.spring(response: 0.3, dampingFraction: 0.85), value: store.isSpotMode)
        }
        .overlay(alignment: .center) {
            if isTargeted {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .strokeBorder(Palette.accent, style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                    .padding(18)
                    .allowsHitTesting(false)
            }
        }
        .onDrop(of: [.image, .fileURL], isTargeted: $isTargeted) { providers in
            store.handleDrop(providers: providers)
        }
    }

}

private struct EmptyCanvas: View {
    let isTargeted: Bool

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "square.and.arrow.down.on.square")
                .font(.system(size: 44, weight: .light))
                .foregroundStyle(isTargeted ? Palette.accent : Color.secondary.opacity(0.6))
            Text("Drop a TLC plate photo")
                .font(.system(size: 16, weight: .semibold))
            Text("or click the + box in the toolbar below")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
    }
}
