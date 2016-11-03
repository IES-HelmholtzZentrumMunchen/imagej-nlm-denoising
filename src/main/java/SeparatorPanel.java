import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * A separator panel for dialogs.
 * @author Julien Pontabry
 */
public class SeparatorPanel extends Panel {
    /**
     * Title of the panel.
     */
    private String m_title;

    /**
     * Insets of the panel.
     */
    private Insets insets = new Insets( 10, 100, 10, 100 );

    /**
     * Default constructor.
     */
    public SeparatorPanel() {
        this("");
    }

    /**
     * Constructor.
     * @param title Title of the panel
     */
    public SeparatorPanel(String title) {
        m_title = title;
    }

    /**
     * Insets getter.
     * @return The current insets
     */
    public Insets getInsets() {
        return insets;
    }

    /**
     * Paint the panel on the provided graphics.
     * @param g Graphics to paint on.
     */
    public void paint(Graphics g) {
        super.paint(g);

        int width = g.getFontMetrics().stringWidth(m_title);

        g.setColor(this.getForeground());
        g.drawLine(5, 20, 15, 20);
        g.drawLine(width+25, 20, this.getWidth() - 5, 20);
        g.drawString(m_title, 20, 23);
    }

    /**
     * Test main function.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        Frame f = new Frame("SeparatorPanel Tester");

        SeparatorPanel p = new SeparatorPanel("Title of Panel");
        p.add(new Label("Label 1"));
        p.add(new Label("Label 2"));
        p.add(new Label("Label 3"));
        f.add(p);

        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        } );
        f.setBounds(300, 300, 300, 300);
        f.setVisible(true);
    }
}
